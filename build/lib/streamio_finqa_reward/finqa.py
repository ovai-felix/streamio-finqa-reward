"""Answer parsing, reward, and metric for FinQA.

Shared by the eval harness, DPO pair builder, and the GRPO recipe so every
stage scores answers the same way.

Signature matches the Together post-training cookbook:
    reward(prediction: str, target: str) -> float
`target` is the gold answer string from FinQA (e.g. "24.69%", "127.40", "yes").
"""
from __future__ import annotations

import functools
import re

_NUM_RE = re.compile(r"\(?-?\$?\s*-?\d[\d,]*\.?\d*|\(?-?\$?\s*-?\.\d+")

# Exact-match tolerance: 0.5% relative or 0.01 absolute (covers rounding like 127.4 vs 127.40).
EXACT_REL, EXACT_ABS = 0.005, 0.01
# "Close" (partial credit) tolerance for the tiered training reward.
CLOSE_REL = 0.05
# Max relative gap allowed when matching via gold's rounding/truncation.
ROUND_GUARD_REL = 0.15


def extract_final(text: str) -> str | None:
    """Return the text after the last '####', falling back to the last \\boxed{...}."""
    if not text:
        return None
    if "####" in text:
        tail = text.rsplit("####", 1)[1].strip()
        return tail.splitlines()[0].strip() if tail else None
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return boxed[-1].strip() if boxed else None


def parse_value(s: str | None):
    """Parse an answer string into float, 'yes'/'no', or None."""
    if s is None:
        return None
    s = s.replace("\\n", " ").strip().lower()
    if not s:
        return None
    if re.match(r"^(yes|no)\b", s):
        return s[:3].strip()
    m = _NUM_RE.search(s)
    if not m:
        return None
    raw = m.group(0)
    neg = raw.startswith("(") and ")" in s[m.end():m.end() + 2]  # accounting negatives: (123)
    raw = raw.replace("(", "").replace("$", "").replace(",", "").replace(" ", "")
    try:
        v = float(raw)
    except ValueError:
        return None
    return -abs(v) if neg else v


def _within(p: float, g: float, rel: float, abs_tol: float) -> bool:
    # Absolute tolerance only for a zero gold. Applied everywhere, it let "#### 0" match any
    # |gold| <= 1 via the gold/100 candidate (5.1% of GRPO prompts): a reward hack.
    return abs(p - g) <= (abs_tol if g == 0 else rel * abs(g))


def _match(p, g, rel: float, abs_tol: float) -> bool:
    if p is None or g is None:
        return False
    if isinstance(g, str) or isinstance(p, str):
        return p == g
    # FinQA mixes fractions and percents (answer "93.5%" with exe_ans 0.935),
    # so accept the gold value at x1, x100 and /100.
    return any(_within(p, c, rel, abs_tol) for c in (g, g * 100, g / 100))


def _gold_decimals(target: str) -> int | None:
    m = _NUM_RE.search(target.replace("\\n", " "))
    if not m:
        return None
    raw = m.group(0)
    return len(raw.split(".", 1)[1]) if "." in raw else 0


def _match_gold_precision(p, g, target: str) -> bool:
    """Gold is often rounded or truncated (40.3 -> "40%", -68.6 -> "-68%", 7.76 -> "7.8%").
    Accept p if rounding or truncating it to gold's decimal places gives gold, and it is
    within ROUND_GUARD_REL of gold so small integers can't absorb real misses."""
    if not isinstance(p, float) or not isinstance(g, float):
        return False
    d = _gold_decimals(target)
    if d is None:
        return False
    for q in (p, p * 100, p / 100):
        if abs(q - g) > ROUND_GUARD_REL * abs(g):
            continue
        trunc = int(q * 10 ** d) / 10 ** d
        if abs(round(q, d) - g) < 1e-9 or abs(trunc - g) < 1e-9:
            return True
    return False


def _exact(p, g, target: str) -> bool:
    return _match(p, g, EXACT_REL, EXACT_ABS) or _match_gold_precision(p, g, target)


@functools.lru_cache(maxsize=65536)
def finqa_metric(prediction: str, target: str) -> float:
    """Strict 0/1 accuracy. Use this for reporting and validation."""
    return 1.0 if _exact(parse_value(extract_final(prediction)), parse_value(target), target) else 0.0


@functools.lru_cache(maxsize=65536)
def finqa_reward(prediction: str, target: str) -> float:
    """Tiered training reward in [0, 1].

    0.1  followed the '#### <answer>' format with a parseable value
    +0.2 within 5% of gold (gives GRPO a gradient on near-misses)
    +0.7 exact (within 0.5%)
    Empty or runaway outputs score 0.
    """
    if not prediction or not prediction.strip():
        return 0.0
    if len(prediction) > 6000:  # degenerate/rambling completion
        return 0.0
    p, g = parse_value(extract_final(prediction)), parse_value(target)
    if p is None:
        return 0.0
    if _exact(p, g, target):  # exact implies close, even when matched via gold's rounding
        return 1.0
    return 0.3 if _match(p, g, CLOSE_REL, EXACT_ABS) else 0.1
