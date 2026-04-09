# tools/company_scraper.py
#
# Lightweight data fetcher used by agents/researcher.py to enrich
# LLM-discovered companies. Two sources only:
#   1) yfinance — basic financials (revenue, market cap, employees)
#   2) screener.in — corporate announcements, annual reports, concalls,
#      credit ratings (single HTTP fetch parses all four)
# Plus a small PDF helper for annual-report excerpts.

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


class CompanyDataFetcher:
    def __init__(self, company_name: str, yahoo_ticker: Optional[str] = None):
        self.name = company_name
        self.yahoo_ticker = yahoo_ticker

    # ── 1) yfinance financials ───────────────────────────────────────
    def get_financials_from_yahoo(self) -> Dict:
        """Returns clean dict: revenue, market_cap, employees, sector, website."""
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
        }

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
    def extract_annual_report_pdf(self, pdf_url: str) -> str:
        """Download and extract first 3000 characters from PDF to save tokens."""
        resp = requests.get(pdf_url, stream=True, headers=DEFAULT_HEADERS, timeout=30)
        resp.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(resp.content)
        text = ""
        with pdfplumber.open("temp.pdf") as pdf:
            for page in pdf.pages[:3]:  # Only first 3 pages
                text += page.extract_text() or ""
        return text[:3000]
