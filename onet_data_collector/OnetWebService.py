"""Backwards-compatible module for the original ``OnetWebService`` import path.

Historically callers did ``from onet_data_collector.OnetWebService import
OnetWebService``. The implementation now lives in
:mod:`onet_data_collector.client`; this module simply re-exports it so existing
code keeps working.
"""

from __future__ import annotations

from .client import OnetClient, OnetWebService

__all__ = ["OnetClient", "OnetWebService"]
