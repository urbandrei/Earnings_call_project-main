"""T1.2 trading-calendar wrapper (NYSE / XNYS)."""

from datetime import date

from ecvol.data import calendar as cal


def test_known_sessions_and_holidays():
    assert cal.is_session(date(2020, 1, 2))  # first trading day of 2020
    assert not cal.is_session(date(2020, 1, 1))  # New Year's Day
    assert not cal.is_session(date(2021, 7, 4))  # Independence Day (Sunday → closed)
    assert not cal.is_session(date(2020, 1, 4))  # Saturday


def test_session_count_matches_known_window():
    # 2020 had 253 NYSE trading sessions.
    assert cal.session_count(date(2020, 1, 1), date(2020, 12, 31)) == 253


def test_sessions_in_range_is_sorted_and_inclusive():
    sessions = cal.sessions_in_range(date(2020, 1, 2), date(2020, 1, 10))
    assert sessions == sorted(sessions)
    assert sessions[0] == date(2020, 1, 2)
    assert all(cal.is_session(d) for d in sessions)


def test_empty_when_end_before_start():
    assert cal.sessions_in_range(date(2020, 2, 1), date(2020, 1, 1)) == []
