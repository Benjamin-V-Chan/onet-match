"""onet_analysis — the analytics layer of the O*NET career-intelligence pipeline.

Consumes the tidy tables produced by the ``onet_data_collector`` package and
turns them into diagnostics and interpretable structure:

* :mod:`onet_analysis.schema` — resilient column resolution across export variants.
* :mod:`onet_analysis.eda` — coverage stats and distribution plots.
* :mod:`onet_analysis.clustering` — TF-IDF -> TruncatedSVD -> KMeans role clusters.
* :mod:`onet_analysis.keyword_attribution` — where each keyword lands in feature space.
* :mod:`onet_analysis.collect` — thin re-exports of the collector's ETL stages.

The full pipeline (collect -> EDA -> keywords -> cluster) is driven by the
``onet-analyze`` CLI in :mod:`onet_analysis.cli`.
"""

from __future__ import annotations

__version__ = "0.3.1"

__all__ = ["__version__"]
