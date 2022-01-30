import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost


def detect_collision(path1, path2):
    ##############################
    # (DONE 29.01.22) Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    longest_path_length = max(len(path1), len(path2))


    for timestep in range(longest_path_length):
        current_position_1 = get_location(path1, timestep)
        current_position_2 = get_location(path2, timestep)

        previous_position_1 = get_location(path1, timestep-1)
        previous_position_2 = get_location(path2, timestep-2)

        vertex_collision = current_position_1 == current_position_2
        edge_collision = current_position_1 == previous_position_2 and current_position_2 == previous_position_1

        result = {
            "agent_0": {
                "previous_position": previous_position_1,
                "current_position": current_position_1
            },
            "agent_1": {
                "previous_position": previous_position_2,
                "current_position": current_position_2
            },
            "timestep": timestep
        }

        if  vertex_collision:
            result['collision_type'] = 'vertex_collision'
            return result
        elif edge_collision:
            result['collision_type'] = 'edge_collision'
            return result

    return None


def detect_collisions(paths):
    ##############################
    # (DONE 29.01.22) Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    from itertools import combinations

    # Allocate storage & enumerate all agents
    collisions_table = []
    agents = [i for i in range(len(paths))]

    
    # iterate through all pairs of agents
    for agent_0, agent_1 in combinations(agents, 2):
        collision = detect_collision(paths[agent_0], paths[agent_1])
        
        collision['agent_1_id'] = agent_1
        collision['agent_0_id'] = agent_0
        collisions_table.append(collision)

    return collisions_table


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    constraints = []
    collision_type = collision.get('collision_type')

    if collision_type == 'vertex_collision':
        # Create constraints for both agents
        for id in [0,1]:
            constraint = {
                'position': collision.get(f'agent_{id}').get('current_position'),
                'time': collision.get('timestep'),
                'agent': collision.get(f'agent_{id}_id'),
                'type': "vertex_collision"
            }
            constraints.append(constraint)

    elif collision_type == 'edge_collision':
        for id in [0,1]:
            constraint = {
                'to_position': collision.get(f'agent_{id}').get('current_position'),
                'time': collision.get('timestep'),
                'agent': collision.get(f'agent_{id}_id'),
                'type': "edge_collision"
            }
            constraints.append(constraint)
    else:
        raise Exception("Invalid collision type in standard splitting")

    return constraints
    

def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    pass


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        print(root['collisions'])

        # Task 3.2: Testing
        for collision in root['collisions']:
            print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        while self.open_list:
            node = self.pop_node()

            # If there is a collision
            if node.get('collisions'):
                # retrieve first collision
                collision = node.get('collisions')[0]

                # Split the collision for each agent
                constraints = standard_splitting(collision)

                # For each constraint, find a path and continue search
                for constraint in constraints:
                    
                    child = {
                        'cost': 0,
                        'constraints': node.get('constraints') + [constraint],
                        'paths': [],
                        'collisions': []
                    }

                    # Find an updated path for each agent
                    valid_path_found = True
                    for i in range(self.num_of_agents):  
                        path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                                    i, child['constraints'])
                        if path is None:
                            valid_path_found = False
                            # raise BaseException('No solutions')
                        child['paths'].append(path)

                    if valid_path_found:
                        child['cost'] = get_sum_of_cost(child['paths'])
                        child['collisions'] = detect_collisions(child['paths'])
                        
                        self.push_node(child)


            # if there is no collision, return solution
            else:
                return node.paths


        self.print_results(root)
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
