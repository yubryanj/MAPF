import heapq

from collections import namedtuple


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1

            # Disallow invalid transitions
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue

            # Or if it collides into a wall ( note walls encapsulate the agent in these maps)
            if my_map[child_loc[0]][child_loc[1]]:
                continue

            # Create child node
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]

                # Add child back to the heap if the cost is lower than previous costs
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                # If child is never seen before, put it onto the heap
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # (DONE 28.01.22) Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.

    # Allocate storage and retrieve list of unique timesteps
    constraint_table = {}
    constraint_table['vertex_collisions'] = {}
    constraint_table['edge_collisions'] = {}
    constraint_table['termination_collisions'] = {}


    # For each timestep, retrieve occupied positions
    timesteps = list(set([constraint.get('time') for constraint in constraints if constraint.get('agent') == agent]))
    agent_constraints = [constraint for constraint in constraints if constraint.get('agent') == agent]

    vertex_constraints = [constraint for constraint in agent_constraints if constraint.get('type') == 'vertex_constraint']
    edge_constraints = [constraint for constraint in agent_constraints if constraint.get('type') == 'edge_constraint']
    termination_constraints = [constraint for constraint in agent_constraints if constraint.get('type') == 'termination_constraint']

    # Populate the constraint table
    for timestep in timesteps:
        constraint_table['vertex_collisions'][timestep] = [constraint.get('position') for constraint in vertex_constraints if constraint.get('time') == timestep]
        constraint_table['edge_collisions'][timestep] = [constraint.get('edge') for constraint in edge_constraints if constraint.get('time') == timestep]
        constraint_table['termination_collisions'][timestep] = [constraint.get('position') for constraint in termination_constraints if constraint.get('time') == timestep]

    # Record max timestep in constraints
    constraint_table['max_timestep'] = max(timesteps) if timesteps else 0

    return constraint_table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # (DONE 28.01.22) Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step 'time' violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    # Returns True if the movement is invalid
    # Else return False

    # No existing constraints - return immediately
    if constraint_table.get('max_timestep') == 0:
        return False

    # Check for termination across all timesteps
    timesteps = min(next_time, constraint_table.get('max_timestep') + 1)
    termination_collision = next_loc in [position for time in range(timesteps) for position in constraint_table.get('termination_collisions').get(time)]
    if termination_collision:
        return True

    # Check for vertex, and edge collision
    if next_time <= constraint_table.get('max_timestep'):
        vertex_collision = next_loc in constraint_table.get('vertex_collisions').get(next_time)
        edge_collision = (curr_loc, next_loc) in constraint_table.get('edge_collisions').get(next_time)

        if  vertex_collision or \
            edge_collision:
            return True
        
    # To catch loops
    if next_time > constraint_table['max_timestep'] + 1e2:
        return True

    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # (DONE 28.01.22) Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': earliest_goal_timestep}
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root
    
    # Retrieve the constraint table
    constraint_table = build_constraint_table(constraints, agent)

    while len(open_list) > 0:
        curr = pop_node(open_list)
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints

        # If we found the goal
        if curr['loc'] == goal_loc:
            path = get_path(curr)

            # Recheck if this node will collide during its path
            # or if it will terminate in the path of a
            # higher priority agent's path
            valid_path = True
            for timestep in range(len(path)):
                current_location = get_location(path, timestep)
                next_location = get_location(path, timestep + 1)
                if is_constrained(current_location, next_location, timestep + 1, constraint_table):
                    valid_path = False
            
            if valid_path:
                return path

        for dir in range(4):
            # Take a transition
            child_loc = move(curr['loc'], dir)

            # Disallow Transition into a wall
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            
            # If an agent moves into an occupied step in the next timestep
            if is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraint_table):
                continue

            # Generate child node
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'timestep': curr['timestep'] + 1
                    }

            # If child node cost is lower than previous cost, 
            # or if it has never been seen before
            # add it to the heap
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions
