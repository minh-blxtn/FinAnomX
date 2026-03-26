from __future__ import annotations

from datetime import datetime, timedelta


def parse_ymd(s: str) -> datetime:
    # Expected: YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d")


def ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def make_window(center: str, left: int, right: int) -> tuple[str, str]:
    c = parse_ymd(center)
    return ymd(c - timedelta(days=left)), ymd(c + timedelta(days=right))