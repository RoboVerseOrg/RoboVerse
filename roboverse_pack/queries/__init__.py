"""Queries for RoboVerse. Basic queries from metasim will be automatically imported."""

import metasim.queries as metasim_queries
from metasim.queries import *

__all__ = [
    *metasim_queries.__all__,
]
