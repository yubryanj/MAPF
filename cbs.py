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
        current_position_0 = get_location(path1, timestep)
        current_position_1 = get_location(path2, timestep)

        next_position_0 = get_location(path1, timestep + 1)
        next_position_1 = get_location(path2, timestep + 1)

        vertex_collision = next_position_0 == next_position_1
        edge_collision = current_position_1 == next_position_0 and current_position_0 == next_position_1

        if vertex_collision:
            # Both agent 0 and agent 1 will both be in next_position_1 at time t+1
            return {
                "time": timestep + 1,
                "position": next_position_1,
                "agents": None,
                "type": 'vertex_collision'
            }
        elif edge_collision:
            # Agent 0, in current_position_0 is moving into next_position_0 at time t
            # Agent 1, in current_position_1 is moving into next_position_1 at time t
            # However, current_position_0 is next_position_1 and next_position_0 is current_position_1
            # Thus an illegal swap of position occurs
            return {
                "time": timestep,
                "edge": (current_position_0, next_position_0),
                "agents": None,
                "type": 'edge_collision'
            }

    return None


def detect_collisions(paths):
    ##############################
    # (DONE 29.01.22) Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    from itertools import combinations

    # Allocate storage & enumerate all agents
    first_collisions = []
    agents = [i for i in range(len(paths))]

    # iterate through all pairs of agents
    for agent_0, agent_1 in combinations(agents, 2):
        collision = detect_collision(paths[agent_0], paths[agent_1])
        if collision:
            collision['agents'] = (agent_0, agent_1)
            first_collisions.append(collision)

    return first_collisions


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
    collision_type = collision.get('type')

    if collision_type == 'vertex_collision':
        # Create constraints for both agents
        for agent in collision.get('agents'):
            constraint = {
                'time': collision.get('time'),
                'position': collision.get('position'),
                'agent': agent,
                'type': "vertex_constraint"
            }
            constraints.append(constraint)

    elif collision_type == 'edge_collision':
        agent_0, agent_1 = collision.get('agents')
        constraint_0 = {
            'time': collision.get('time'),
            'edge': collision.get('edge'),
            'agent': agent_0,
            'type': "edge_constraint"
        }

        constraint_1 = {
            'time': collision.get('time'),
            'edge': collision.get('edge')[::-1],
            'agent': agent_1,
            'type': "edge_constraint"
        }

        constraints.extend([constraint_0, constraint_1])
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

    constraints = []
    collision_type = collision.get('type')
    
    # Choose agent randomly
    agents = list(collision.get('agents'))
    random.shuffle(agents)
    agent_0, agent_1 = agents


    if collision_type == 'vertex_collision':
        # Create constraints for both agents
        constraint_0 = {
            'time': collision.get('time'),
            'position': collision.get('position'),
            'agent': agent_0,
            'type': "disjoint_vertex_constraint"
        }

        constraint_1 = {
            'time': collision.get('time'),
            'position': collision.get('position'),
            'agent': agent_1,
            'type': "vertex_constraint"
        }
            

    elif collision_type == 'edge_collision':
        constraint_0 = {
            'time': collision.get('time'),
            'edge': collision.get('edge'),
            'agent': agent_0,
            'type': "disjoint_edge_constraint"
        }

        constraint_1 = {
            'time': collision.get('time'),
            'edge': collision.get('edge')[::-1],
            'agent': agent_1,
            'type': "edge_constraint"
        }
    else:
        raise Exception("Invalid collision type in standard splitting")

    constraints.extend([constraint_0, constraint_1])

    return constraints


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
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
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
            if disjoint:
                print(disjoint_splitting(collision))
            else:
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
                if disjoint:
                    constraints = disjoint_splitting(collision)
                else:
                    constraints = standard_splitting(collision)

                # For each constraint, find a path and continue search
                for constraint in constraints:
                    
                    # Update constraints
                    if constraint in node.get('constraints'):
                        new_constraints = node.get('constraints')
                    else:
                        new_constraints = node.get('constraints') + [constraint]

                    # Prepare child node
                    child = {
                        'cost': 0,
                        'constraints': new_constraints,
                        'paths': [],
                        'collisions': []
                    }

                    # Find an updated path for each agent
                    valid_paths_found = True
                    for i in range(self.num_of_agents):  
                        path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                                    i, child['constraints'])
                        if path is None:
                            valid_paths_found = False
                            # raise BaseException('No solutions')
                        child['paths'].append(path)

                    if valid_paths_found:
                        child['cost'] = get_sum_of_cost(child['paths'])
                        child['collisions'] = detect_collisions(child['paths'])
                        self.push_node(child)

            # if there is no collision, return solution
            else:
                self.print_results(root)
                return node.get('paths')



    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
