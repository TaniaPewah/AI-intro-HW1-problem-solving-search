from framework.graph_search import *
from .map_problem import MapProblem, MapState


class AirDistHeuristic(HeuristicFunction):
    heuristic_name = 'AirDist'

    def estimate(self, state: GraphProblemState) -> float:
        """
        The air distance between the geographic location represented
         by `state` and the geographic location of the problem's target.

        TODO [Ex.11]: implement this method!
        Use `self.problem` to access the problem.
        Use `self.problem.streets_map` to access the map.
        Given a junction index, use `streets_map[junction_id]` to find the
         junction instance (of type `Junction`).
        Use the method `calc_air_distance_from()` to calculate the air
         distance between two junctions.
        """
        assert isinstance(self.problem, MapProblem)
        assert isinstance(state, MapState)

        target_id = self.problem.target_junction_id
        target_junc = self.problem.streets_map[target_id]

        return self.problem.streets_map[state.junction_id].calc_air_distance_from(target_junc)
