"""Trading-calendar helpers (NYSE / XNYS) via `exchange_calendars` (T1.2 subtask).

A thin, cached wrapper so the rest of the pipeline never touches
`exchange_calendars` directly. T1.2 uses it to judge price-series completeness
(how many trading sessions *should* a ticker have between its first and last
quote); T1.3/T1.6 reuse it for the after-hours date rule and embargo logic.
"""

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

CALENDAR = "XNYS"  # NYSE; FinCall + MAEC are US-listed equities


@lru_cache(maxsize=1)
def _calendar() -> xcals.ExchangeCalendar:
    return xcals.get_calendar(CALENDAR)


def is_session(day: date) -> bool:
    """True if `day` is a NYSE trading session."""
    return _calendar().is_session(pd.Timestamp(day))


def sessions_in_range(start: date, end: date) -> list[date]:
    """Trading sessions in the inclusive [start, end] interval, ascending."""
    if end < start:
        return []
    sessions = _calendar().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [ts.date() for ts in sessions]


def session_count(start: date, end: date) -> int:
    """Number of NYSE sessions in the inclusive [start, end] interval."""
    return len(sessions_in_range(start, end))
