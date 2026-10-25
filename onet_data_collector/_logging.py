"""Lightweight logging setup shared across the package.

The library itself never configures the root logger (that is the application's
job). :func:`get_logger` just returns a namespaced logger; :func:`configure`
is a convenience for CLI/script use that attaches a simple stream handler.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "onet_data_collector"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under ``onet_data_collector``."""
    if name is None or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def configure(level: int | str = logging.INFO) -> None:
    """Attach a single stream handler to the package logger (idempotent).

    Intended for CLIs and ad-hoc scripts. Libraries importing this package
    should configure logging themselves instead of calling this.
    """
    logger = get_logger()
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.propagate = False
