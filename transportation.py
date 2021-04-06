"""
Transportation Problem
Author:
Aditya Girish Pawate
18MA20054
Submission of OR lab-9
"""
import numpy as np
import pandas as pd
import math

INFINITY = 10e9


# This function takes input
def take_input():
    # This takes input of supply as a python list
    supply = [float(i) for i in input().split(" ")]
    # This takes input of demand as a python list
    demand = [float(i) for i in input().split(" ")]
    # This is the cost matrix for transportation
    cost_for_transportation = []
    for i in range(len(supply)):
        row_cost = [float(i) for i in input().split(" ")]
        cost_for_transportation.append(row_cost)
    return supply, demand, np.array(cost_for_transportation)


# This function makes sure that it is a balanced transportation problem
def balanced_transportation(supply, demand, cost_for_transportation):
    total_supply = sum(supply)
    total_demand = sum(demand)
    if total_supply < total_demand:
        penalties = np.zeros((1, len(demand)))
        supply.append(total_demand - total_supply)
        new_supply = supply
        new_costs = np.concatenate([cost_for_transportation, penalties], axis=0)
        return new_supply, demand, new_costs
    if total_supply > total_demand:
        demand.append(total_supply - total_demand)
        new_demand = demand
        new_costs = np.concatenate([cost_for_transportation, np.zeros((len(supply), 1))], axis=1)
        return supply, new_demand, new_costs
    return supply, demand, cost_for_transportation


# This is the implementation of North West corner method
def north_west_corner(supply, demand):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    i = 0
    j = 0
    basic_feasible_solutions = []
    while len(basic_feasible_solutions) < len(supply) + len(demand) - 1:
        s = supply_copy[i]
        d = demand_copy[j]
        v = min(s, d)
        supply_copy[i] -= v
        demand_copy[j] -= v
        basic_feasible_solutions.append(((i, j), v))
        if supply_copy[i] == 0 and i < len(supply) - 1:
            i += 1
        elif demand_copy[j] == 0 and j < len(demand) - 1:
            j += 1
    return basic_feasible_solutions


# This function gets all the u and v values
def initialize_us_and_vs(basic_feasible_solutions, costs):
    us = [None] * len(costs)
    vs = [None] * len(costs[0])
    us[0] = 0
    bfs_copy = basic_feasible_solutions.copy()
    while len(bfs_copy) > 0:
        for index, bv in enumerate(bfs_copy):
            i, j = bv[0]
            if us[i] is None and vs[j] is None:
                continue
            cost = costs[i][j]
            if us[i] is None:
                us[i] = cost - vs[j]
            else:
                vs[j] = cost - us[i]
            bfs_copy.pop(index)
            break
    return us, vs


# This function finds the values of w_i
def initialize_ws(basic_feasible_solutions, costs, us, vs):
    ws = []
    for i, row in enumerate(costs):
        for j, cost in enumerate(row):
            non_basic = all([p[0] != i or p[1] != j for p, v in basic_feasible_solutions])
            if non_basic:
                ws.append(((i, j), us[i] + vs[j] - cost))
    return ws


# This function finds whether the ws can be improved
def can_be_improved(ws):
    for p, v in ws:
        if v > 0: return True
    return False


# This function is to get entering variable position
def get_entering_variable_position(ws):
    ws_copy = ws.copy()
    ws_copy.sort(key=lambda w: w[1])
    return ws_copy[-1][0]


# This function is to get the next nodes
def get_possible_next_nodes(loop, not_visited):
    last_node = loop[-1]
    nodes_in_row = [n for n in not_visited if n[0] == last_node[0]]
    nodes_in_column = [n for n in not_visited if n[1] == last_node[1]]
    if len(loop) < 2:
        return nodes_in_row + nodes_in_column
    else:
        prev_node = loop[-2]
        row_move = prev_node[0] == last_node[0]
        if row_move: return nodes_in_column
        return nodes_in_row


# This function gets the loop
def get_loop(basic_variable_position, entering_variable_position):
    def inner_function(loop):
        if len(loop) > 3:
            can_be_closed = len(get_possible_next_nodes(loop, [entering_variable_position])) == 1
            if can_be_closed: return loop

        not_visited = list(set(basic_variable_position) - set(loop))
        possible_next_nodes = get_possible_next_nodes(loop, not_visited)
        for next_node in possible_next_nodes:
            new_loop = inner_function(loop + [next_node])
            if new_loop: return new_loop
    return inner_function([entering_variable_position])


# This function gets the loop pivoting
def loop_pivoting(basic_feasible_solutions, loop):
    even_cells = loop[0::2]
    odd_cells = loop[1::2]
    get_bv = lambda pos: next(v for p, v in basic_feasible_solutions if p == pos)
    leaving_position = sorted(odd_cells, key=get_bv)[0]
    leaving_value = get_bv(leaving_position)
    new_bfs = []
    for p, v in [bv for bv in basic_feasible_solutions if bv[0] != leaving_position] + [(loop[0], 0)]:
        if p in even_cells:
            v += leaving_value
        elif p in odd_cells:
            v -= leaving_value
        new_bfs.append((p, v))
    return new_bfs


# The 
def transportation_solution(supply, demand, costs):
    balanced_supply, balanced_demand, balanced_costs = balanced_transportation(supply, demand, costs)
    count = 0
    def inner_function(basic_feasible_solutions, count):
        us, vs = initialize_us_and_vs(basic_feasible_solutions, balanced_costs)
        ws = initialize_ws(basic_feasible_solutions, balanced_costs, us, vs)
        if can_be_improved(ws):
            count += 1
            entering_variable_position = get_entering_variable_position(ws)
            loop = get_loop([p for p, v in basic_feasible_solutions], entering_variable_position)
            print("Iteration", count, " The loop is ", loop)
            print("The entering variable position is:", entering_variable_position)
            return inner_function(loop_pivoting(basic_feasible_solutions, loop), count)
        return basic_feasible_solutions, count

    basic_variables, count = inner_function(north_west_corner(balanced_supply, balanced_demand), count)
    solution = np.zeros((len(balanced_costs), len(balanced_costs[0])))
    for (i, j), v in basic_variables:
        solution[i][j] = v
    return solution


def get_total_cost(costs, solution):
    total_cost = 0
    for i, row in enumerate(costs):
        for j, cost in enumerate(row):
            total_cost += cost * solution[i][j]
    return total_cost


if __name__ == '__main__':
    supply, demand, cost_for_transportation = take_input()
    solution = transportation_solution(supply, demand, cost_for_transportation)
    print("The final solution for each variable is")
    print(solution)
    print('total cost: ', get_total_cost(cost_for_transportation, solution))
