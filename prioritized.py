import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = []

        for i in range(self.num_of_agents):  # Find path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            ##############################
            # Task 2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches
            lower_priority_agents = [j for j in range(i+1,self.num_of_agents)]
            for agent in lower_priority_agents:
                for time, position in enumerate(path):

                    # Lower priority agents cannot be on this vertex at time t
                    vertex_constraint = {
                        "time": time,
                        "position": position,
                        "agent": agent,
                        "type": 'vertex_constraint'
                    }

                    # Lower priority agents cannot do the inverse transition 
                    # from position path[t] to path[t-1] at time t
                    edge_constraint = {
                        "time": time - 1,
                        "edge": (path[time], path[time-1]), # Lower priority agents cannot do the reverse of this transition
                        "agent": agent,
                        "type": "edge_constraint"
                    }

                    constraints.append(vertex_constraint)
                    constraints.append(edge_constraint)

                # Other agents cannot path through a vertex
                # where an agent has terminated their path
                termination_constraint = {
                    "time": time,
                    "position": path[-1],
                    "agent": agent,
                    "type": "termination_constraint"
                }
                constraints.append(termination_constraint)
                
            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
