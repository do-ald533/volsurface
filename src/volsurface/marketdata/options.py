"""Plain option records shared by the provider factory and the provider implementations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IbkrOptions:
    """Connection, subscription limits and market inputs for the Interactive Brokers provider.

    The broker feed carries no interest rate or dividend yield, so both are supplied here.
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 35
    max_lines: int = 80
    expiries: int = 6
    band: float = 0.08
    min_days: int = 3
    max_days: int = 120
    market_data_type: int = 3
    connect_timeout_s: float = 10.0
    rate: float = 0.04
    dividend_yield: float = 0.012
