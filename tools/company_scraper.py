# tools/company_scraper.py
#
# Lightweight data fetcher used by agents/researcher.py to enrich
# LLM-discovered companies. Two sources only:
#   1) yfinance — basic financials (revenue, market cap, employees, margins,
#      growth, debt, target price, business summary)
#   2) screener.in — corporate announcements, annual reports, concalls,
#      credit ratings (single HTTP fetch parses all four)
# Plus a robust PDF helper for annual-report excerpts.

import os
import tempfile
import requests
import pdfplumber
from bs4 import BeautifulSoup
from typing import Optional, Dict, List

import yfinance as yf


SCREENER_BASE = "https://www.screener.in/company"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

PDF_MAX_BYTES = 30 * 1024 * 1024  # 30 MB


class CompanyDataFetcher:
    def __init__(self, company_name: str, yahoo_ticker: Optional[str] = None):
        self.name = company_name
        self.yahoo_ticker = yahoo_ticker

    # ── 1) yfinance financials ───────────────────────────────────────
    def get_financials_from_yahoo(self) -> Dict:
        """Returns expanded yfinance projection. Raw INR values; researcher
        converts to crore where the unified schema requires it."""
        if not self.yahoo_ticker:
            return {"error": "No ticker"}
        ticker = yf.Ticker(self.yahoo_ticker)
        info = ticker.info or {}
        return {
            "revenue": info.get("totalRevenue"),
            "market_cap": info.get("marketCap"),
            "employees": info.get("fullTimeEmployees"),
            "sector": info.get("sector"),
            "website": info.get("website"),
            "ebitda_margins": info.get("ebitdaMargins"),
            "profit_margins": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "free_cashflow": info.get("freeCashflow"),
            "total_debt": info.get("totalDebt"),
            "current_price": info.get("currentPrice"),
            "target_mean_price": info.get("targetMeanPrice"),
            "recommendation_key": info.get("recommendationKey"),
            "long_business_summary": info.get("longBusinessSummary"),
            "industry": info.get("industry"),
        }

    def get_quarterly_revenue(self) -> List[Dict]:
        """Real quarterly revenue from yfinance. Returns last 4 quarters as
        [{quarter, revenue_crore, source:'yfinance'}]. Most-recent first.
        Empty list on any failure or missing data — caller decides fallback.
        Replaces the LLM's hallucinated revenue_quarters in the researcher."""
        if not self.yahoo_ticker:
            return []
        try:
            ticker = yf.Ticker(self.yahoo_ticker)
            qf = ticker.quarterly_financials
            if qf is None or qf.empty or "Total Revenue" not in qf.index:
                return []
            revenue_row = qf.loc["Total Revenue"]
            items: List[Dict] = []
            for date, value in revenue_row.items():
                if value is None:
                    continue
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                if val != val:  # NaN check without importing math
                    continue
                quarter_label = (
                    date.strftime("%b %Y") if hasattr(date, "strftime") else str(date)
                )
                items.append({
                    "quarter": quarter_label,
                    "revenue_crore": round(val / 1e7, 2),  # raw INR → crore
                    "source": "yfinance",
                })
            return items[:4]
        except Exception:
            return []

    # ── 2) screener.in documents ─────────────────────────────────────
    def get_screener_documents(self, screener_symbol: str) -> Dict:
        """
        Scrape https://www.screener.in/company/{symbol}/ and return:
            {
              "announcements":   [{title, url, date, context?}, ...],
              "annual_reports":  [{title, url, date}, ...],
              "concalls":        [{title, url, date?}, ...],
              "credit_ratings":  [{title, url, date?}, ...],
              "url":             "<page url>",
            }

        screener_symbol is the NSE symbol (e.g. 'SONATSOFTW') or BSE scrip
        code — screener.in accepts both. Sub-lists are independent; any
        one may be empty.
        """
        url = f"{SCREENER_BASE}/{screener_symbol}/"
        empty = {
            "announcements": [],
            "annual_reports": [],
            "concalls": [],
            "credit_ratings": [],
            "url": url,
        }
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            empty["error"] = f"fetch failed: {e}"
            return empty

        soup = BeautifulSoup(resp.text, "html.parser")
        section = soup.find("section", id="documents")
        if not section:
            empty["error"] = "documents section not found on page"
            return empty

        return {
            "announcements":  self._parse_doc_block(section, kind="announcements"),
            "annual_reports": self._parse_doc_block(section, kind="annual-reports"),
            "concalls":       self._parse_doc_block(section, kind="concalls"),
            "credit_ratings": self._parse_doc_block(section, kind="credit-ratings"),
            "url": url,
        }

    @staticmethod
    def _parse_doc_block(section, kind: str) -> List[Dict]:
        """
        Find <div class="documents {kind} flex-column"> inside the section
        and return its <ul class="list-links"> entries as dicts.

        For 'announcements' the wrapper is the BARE 'documents flex-column'
        (no second class word), so we special-case it.
        """
        if kind == "announcements":
            block = section.find(
                "div",
                class_=lambda c: bool(
                    c
                    and "documents" in c.split()
                    and "flex-column" in c.split()
                    and len(c.split()) == 2  # bare 'documents flex-column'
                ),
            )
        else:
            block = section.find("div", class_=f"documents {kind} flex-column")

        if not block:
            return []

        items: List[Dict] = []
        for li in block.select("ul.list-links li"):
            a = li.find("a")
            if not a or not a.get("href"):
                continue

            href = a["href"]
            # Inner <span> = bare date, inner <div> = date + context summary
            date_el = a.find(
                ["span", "div"],
                class_=lambda c: bool(c and "ink-600" in c),
            )
            date_text = date_el.get_text(" ", strip=True) if date_el else ""

            # Title = full anchor text minus the date_el text
            full = a.get_text(" ", strip=True)
            title = full.replace(date_text, "").strip(" -—") if date_text else full

            entry: Dict = {"title": title, "url": href}

            # Split "1 Apr - Rajsekhar Datta Roy CDO; ..." → date + context
            if " - " in date_text:
                date, _, context = date_text.partition(" - ")
                entry["date"] = date.strip()
                entry["context"] = context.strip()
            else:
                entry["date"] = date_text

            items.append(entry)

        return items

    # ── 3) Annual report PDF excerpt ─────────────────────────────────
    def extract_annual_report_pdf(self, pdf_url: str) -> Optional[str]:
        """Download and extract MD&A-section text. Returns None on any failure
        rather than raising — annual reports are best-effort enrichment.
          - unique tmp file per company (no race, no overwrite)
          - skip if Content-Type isn't pdf
          - skip if Content-Length > 30 MB
          - extract pages 8-19 (skip cover/index, hit MD&A)
          - fallback to first 5 pages if PDF has fewer than 8 pages
          - per-page try/except so one bad page doesn't kill the whole pull
          - always cleans up the tmp file
        """
        safe_name = (self.name or "ar")[:15].replace(" ", "_")
        tmp = tempfile.NamedTemporaryFile(
            suffix=".pdf",
            prefix=f"ar_{safe_name}_",
            delete=False,
        )
        tmp_path = tmp.name
        tmp.close()

        try:
            resp = requests.get(
                pdf_url, stream=True, headers=DEFAULT_HEADERS, timeout=30
            )
            resp.raise_for_status()

            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "pdf" not in ctype:
                return None

            clen = resp.headers.get("Content-Length")
            if clen:
                try:
                    if int(clen) > PDF_MAX_BYTES:
                        return None
                except ValueError:
                    pass

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            text = ""
            with pdfplumber.open(tmp_path) as pdf:
                num_pages = len(pdf.pages)
                page_range = pdf.pages[8:20] if num_pages >= 8 else pdf.pages[:5]
                for page in page_range:
                    try:
                        text += (page.extract_text() or "")
                    except Exception:
                        continue
            return text[:3000] if text else None
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
