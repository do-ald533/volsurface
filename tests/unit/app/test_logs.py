import logging

import pytest

from volsurface.app.logs import (
    CHATTY_LOGGERS,
    BrokerNoticeFilter,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _restore_logging() -> object:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    levels = {name: logging.getLogger(name).level for name in CHATTY_LOGGERS}
    filters = {name: list(logging.getLogger(name).filters) for name in CHATTY_LOGGERS}
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    for name, previous in levels.items():
        logging.getLogger(name).setLevel(previous)
    for name, previous_filters in filters.items():
        logging.getLogger(name).filters[:] = previous_filters


def test_broker_notices_are_quiet_unless_debugging() -> None:
    configure_logging("INFO")
    for name in CHATTY_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING
    configure_logging("DEBUG")
    for name in CHATTY_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() == logging.DEBUG


def test_application_loggers_are_unaffected(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    get_logger("volsurface.test").info("hello", answer=42)
    assert "hello" in capsys.readouterr().err


def test_json_output_is_one_object_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json=True)
    get_logger("volsurface.test").info("hello", answer=42)
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert line.startswith("{") and '"answer": 42' in line


def _record(message: str, level: int = logging.ERROR) -> logging.LogRecord:
    return logging.LogRecord("ib_async.wrapper", level, __file__, 1, message, None, None)


class TestBrokerNoticeFilter:
    DELAYED = (
        "Error 10091, reqId %d: Part of requested market data requires additional subscription "
        "for API. Delayed market data is available."
    )

    def test_a_repeated_notice_is_shown_once_as_info_and_then_hidden(self) -> None:
        notice = BrokerNoticeFilter()
        first = _record(self.DELAYED % 15)
        assert notice.filter(first) is True
        assert first.levelno == logging.INFO and first.levelname == "INFO"
        assert "repeats of code 10091 are hidden" in first.getMessage()
        for req_id in range(16, 90):
            assert notice.filter(_record(self.DELAYED % req_id)) is False
        assert notice.repeats == {10091: 74}

    def test_real_errors_are_never_hidden(self) -> None:
        notice = BrokerNoticeFilter()
        for message in (
            "Error 200, reqId 17: No security definition has been found for the request",
            "Error 10168, reqId 5: Requested market data is not subscribed.",
            "Error 354, reqId 5: Not subscribed to requested market data.",
            "Connecting to 127.0.0.1:7497 with clientId 35...",
        ):
            first, again = _record(message), _record(message)
            assert notice.filter(first) is True and notice.filter(again) is True
            assert first.levelno == logging.ERROR

    def test_each_notice_code_gets_its_own_first_line(self) -> None:
        notice = BrokerNoticeFilter()
        assert notice.filter(_record("Warning 2104, reqId -1: farm OK:usfarm", logging.INFO))
        assert notice.filter(_record("Warning 2106, reqId -1: HMDS OK:ushmds", logging.INFO))
        assert not notice.filter(_record("Warning 2104, reqId -1: farm OK:eufarm", logging.INFO))

    def test_the_filter_is_installed_unless_debugging_and_never_stacked(self) -> None:
        wrapper = logging.getLogger("ib_async.wrapper")
        configure_logging("INFO")
        configure_logging("INFO")
        assert len([f for f in wrapper.filters if isinstance(f, BrokerNoticeFilter)]) == 1
        configure_logging("DEBUG")
        assert not [f for f in wrapper.filters if isinstance(f, BrokerNoticeFilter)]
