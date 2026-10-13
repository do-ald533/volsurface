"""Structured logging: structlog on top of the standard library.

Both ``structlog`` loggers and plain ``logging`` loggers (used by the event bus and third-party
libraries) end up in one handler, rendered either for a terminal or as JSON lines.
"""

import logging
import re
import sys
from typing import Any

import structlog

CHATTY_LOGGERS = ("ib_async.wrapper",)
NOTICE_CODES = frozenset({2104, 2106, 2107, 2108, 2158, 2186, 10091, 10167})
_BROKER_CODE = re.compile(r"^(?:Error|Warning) (\d+),")


class BrokerNoticeFilter(logging.Filter):
    """Show each informational broker notice once, as INFO, and count the repeats.

    Interactive Brokers repeats notices such as "delayed market data is available" (code 10091)
    once per subscribed contract, and ``ib_async`` logs some of them as errors. They describe data
    that does arrive, so one line per code is enough.
    """

    def __init__(self) -> None:
        super().__init__()
        self.repeats: dict[int, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        match = _BROKER_CODE.match(record.getMessage())
        if match is None or int(match.group(1)) not in NOTICE_CODES:
            return True
        code = int(match.group(1))
        if code in self.repeats:
            self.repeats[code] += 1
            return False
        self.repeats[code] = 0
        record.msg = f"{record.getMessage()} (repeats of code {code} are hidden)"
        record.args = ()
        record.levelno = logging.INFO
        record.levelname = "INFO"
        return True


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Install the root handler. Safe to call more than once.

    Interactive Brokers sends one informational notice per subscribed contract (``ib_async``
    logs them at INFO), which buries everything else, so that logger only reports warnings and
    errors unless the level is DEBUG.
    """
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%H:%M:%S" if not json else "iso", utc=json),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in CHATTY_LOGGERS:
        chatty = logging.getLogger(name)
        chatty.setLevel(logging.NOTSET if level == "DEBUG" else logging.WARNING)
        for old in [f for f in chatty.filters if isinstance(f, BrokerNoticeFilter)]:
            chatty.removeFilter(old)
        if level != "DEBUG":
            chatty.addFilter(BrokerNoticeFilter())


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A structlog logger bound to ``name``."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
