"""Map free text about work to O*NET occupations.

    from onet_match import match
    result = match("ran the robotics club's CAD team", top_k=5)
"""

from onet_analysis import __version__
from onet_analysis.matching import MatchEngine, MatchResult, OccupationMatch, default_engine, match

__all__ = ["match", "MatchEngine", "MatchResult", "OccupationMatch", "default_engine", "__version__"]
