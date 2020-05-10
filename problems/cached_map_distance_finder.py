from typing import *

from framework import *
from .map_problem import MapProblem


class CachedMapDistanceFinder:
    """
    This is a helper class, used to find distances in the map and cache distances that has already been calculated.
    Calculating a distance (between 2 junctions) in the map is performed by solving a `MapProblem` using a
     `GraphProblemSolver`. `CachedMapDistanceFinder` receives the solver to use in its c'tor.
    """

    def __init__(self, streets_map: StreetsMap, map_problem_solver: GraphProblemSolver):
        self.streets_map = streets_map
        self.map_problem_solver = map_problem_solver

        self._cache: Dict[Tuple[int, int], Optional[Cost]] = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key: Tuple[int, int], val: Optional[Cost]):
        self._cache[key] = val

    def _get_from_cache(self, key: Tuple[int, int]) -> Optional[Cost]:
        return self._cache.get(key)

    def _is_in_cache(self, key: Tuple[int, int]) -> bool:
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return key in self._cache

    def get_map_cost_between(self, src_junction: Junction, tgt_junction: Junction) -> Optional[Cost]:
        """
        [Ex.13]:
        """
        key = (src_junction.id, tgt_junction.id)
        if self._is_in_cache(key):
            return _get_from_cache(key)
        map_problem = MapProblem(self.streets_map, src_junction.id, tgt_junction.id)
        res = self.map_problem_solver.solve_problem(map_problem)
        self._insert_to_cache(key, res)
        return res
