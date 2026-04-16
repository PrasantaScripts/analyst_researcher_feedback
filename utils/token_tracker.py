# utils/token_tracker.py — Tracks token usage across all agents.
# Import the global `tracker` instance and call tracker.record() after each agent response.

import time
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class CallRecord:
    agent: str
    timestamp: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_sec: float
    prompt_preview: str  # first 80 chars


class TokenTracker:
    """Singleton tracker — import and use the global `tracker` instance."""

    def __init__(self):
        self.calls: list[CallRecord] = []
        self.start_time: float | None = None

    def reset(self):
        self.calls = []
        self.start_time = time.time()

    # Record one API call
    def record(self, agent_name: str, response, prompt: str, latency: float):
        """Extract token counts from a Strands Agent response and store them.
        Tries multiple paths because the Strands SDK exposes token info differently
        across versions (response.metrics, response.usage, response.result.usage)."""
        inp, out = 0, 0

        # Path 1: response.metrics dict (strands-agents >=0.1.6)
        metrics = getattr(response, "metrics", None) or {}
        if isinstance(metrics, dict):
            inp = metrics.get("inputTokens", metrics.get("input_tokens", 0))
            out = metrics.get("outputTokens", metrics.get("output_tokens", 0))

        # Path 2: response.usage (some SDK versions)
        if inp == 0 and out == 0:
            usage = getattr(response, "usage", None) or {}
            if isinstance(usage, dict):
                inp = usage.get("input_tokens", usage.get("inputTokens", 0))
                out = usage.get("output_tokens", usage.get("outputTokens", 0))

        # Path 3: look in response.result or response.message
        if inp == 0 and out == 0:
            for attr in ("result", "message", "raw"):
                inner = getattr(response, attr, None)
                if inner and hasattr(inner, "usage"):
                    u = inner.usage
                    if isinstance(u, dict):
                        inp = u.get("input_tokens", 0)
                        out = u.get("output_tokens", 0)
                    elif hasattr(u, "input_tokens"):
                        inp = getattr(u, "input_tokens", 0)
                        out = getattr(u, "output_tokens", 0)

        total = inp + out

        # Estimate from text length if SDK doesn't expose counts
        if total == 0:
            resp_text = str(response)
            inp = max(len(prompt) // 4, 1)  # rough ~4 chars/token
            out = max(len(resp_text) // 4, 1)
            total = inp + out

        rec = CallRecord(
            agent=agent_name,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            latency_sec=round(latency, 2),
            prompt_preview=prompt[:80].replace("\n", " "),
        )
        self.calls.append(rec)

        # Live mini-log
        print(f"   Tokens -> in: {inp:,}  out: {out:,}  total: {total:,}  ({latency:.1f}s)")

    # Summary helpers
    def totals(self):
        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        return total_input, total_output, total_input + total_output

    def by_agent(self) -> dict[str, dict]:
        agents: dict[str, dict] = {}
        for c in self.calls:
            a = agents.setdefault(
                c.agent,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total": 0, "latency": 0.0},
            )
            a["calls"] += 1
            a["input_tokens"] += c.input_tokens
            a["output_tokens"] += c.output_tokens
            a["total"] += c.total_tokens
            a["latency"] += c.latency_sec
        return agents

    def estimate_cost(self, price_per_1k_input=0.003, price_per_1k_output=0.015) -> float:
        """Rough cost estimate — adjust prices for your Bedrock model."""
        total_input, total_output, _ = self.totals()
        return (total_input / 1000) * price_per_1k_input + (total_output / 1000) * price_per_1k_output

    # Console report
    def print_summary(self):
        elapsed = time.time() - (self.start_time or time.time())
        total_input, total_output, total = self.totals()
        cost = self.estimate_cost()

        print("\n")
        print("╔" + "═" * 62 + "╗")
        print("║" + " TOKEN USAGE SUMMARY".center(62) + "║")
        print("╠" + "═" * 62 + "╣")

        # Per-agent breakdown
        for name, stats in self.by_agent().items():
            print(f"║  {name:<16} │ calls: {stats['calls']:<3} │ "
                  f"in: {stats['input_tokens']:>7,} │ out: {stats['output_tokens']:>7,} ║")

        print("╠" + "═" * 62 + "╣")
        print(f"║  {'TOTAL':<16} │ calls: {len(self.calls):<3} │ "
              f"in: {total_input:>7,} │ out: {total_output:>7,} ║")
        print(f"║  {'Grand total tokens:':<30} {total:>15,}        ║")
        print(f"║  {'Estimated cost (USD):':<30} ${cost:>14.4f}        ║")
        print(f"║  {'Pipeline duration:':<30} {elapsed:>12.0f}s        ║")
        print("╚" + "═" * 62 + "╝")

        # Call-by-call detail
        if self.calls:
            print("\n── Call-by-call detail ─────────────────────────────────────────")
            print(f"  {'#':>3}  {'Agent':<16} {'Time':>8}  {'In':>7}  {'Out':>7}  {'Lat':>6}  Preview")
            print(f"  {'─'*3}  {'─'*16} {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*30}")
            for i, c in enumerate(self.calls, 1):
                print(f"  {i:>3}  {c.agent:<16} {c.timestamp:>8}  "
                      f"{c.input_tokens:>7,}  {c.output_tokens:>7,}  "
                      f"{c.latency_sec:>5.1f}s  {c.prompt_preview[:30]}…")
        print()


# Global instance — import this everywhere
tracker = TokenTracker()
