#!/usr/bin/env python3
import time
from collections import deque

from Agent import *  # See the Agent.py file
from pysat.solvers import Glucose3


# All your code can go here.

# You can change the main function as you wish. Run this program to see the output. Also see Agent.py code.
def ask_query(kb, q):
    return not kb.solve([-q])


def mine(x, y):
    # return x * 10 + y
    return 50 + x + y * 7


def gold(x, y):
    # return x * 100 + y
    return 1 + x + y * 7


def percept(x, y, k):
    # return x * 100 + y * 10 + k
    return 150 + (x + y * 7) * 4 + k


def visited(x, y):
    # return x * 1000 + y
    return 100 + x + y * 7


def dnf_to_cnf(dnf, i):
    if i == len(dnf) - 1:
        temp = []
        for lit in dnf[i]:
            temp.append([lit])
        # print(temp)
        return temp

    cnf = dnf_to_cnf(dnf, i + 1)
    # print(cnf)
    ans = []
    for ind2 in range(len(dnf[i])):
        for ind in range(len(cnf)):
            temp = cnf[ind].copy()
            temp.append(dnf[i][ind2])
            ans.append(temp)
    return ans


def percept_val_0(i, j):
    dnf = [[-percept(i, j, 0)],
           [-mine(i - 1, j), -mine(i + 1, j), -mine(i, j - 1), -mine(i, j + 1)]]
    # print(dnf)
    cnf = dnf_to_cnf(dnf, 0)
    # print(cnf)
    return cnf


def percept_val_1(i, j):
    dnf = [[-percept(i, j, 1)],
           percept_val_1_helper(mine(i - 1, j), mine(i + 1, j), mine(i, j - 1), mine(i, j + 1)),
           percept_val_1_helper(mine(i + 1, j), mine(i - 1, j), mine(i, j - 1), mine(i, j + 1)),
           percept_val_1_helper(mine(i, j + 1), mine(i + 1, j), mine(i, j - 1), mine(i - 1, j)),
           percept_val_1_helper(mine(i, j - 1), mine(i + 1, j), mine(i - 1, j), mine(i, j + 1))]
    # print(dnf)
    cnf = dnf_to_cnf(dnf, 0)
    # print(cnf)
    return cnf


def percept_val_1_helper(w, x, y, z):
    conjunction = [w, -x, -y, -z]
    return conjunction


def percept_val_2(i, j):
    dnf = [[-percept(i, j, 2)],
           percept_val_2_helper(mine(i - 1, j), mine(i + 1, j), mine(i, j - 1), mine(i, j + 1)),
           percept_val_2_helper(mine(i, j + 1), mine(i - 1, j), mine(i, j - 1), mine(i + 1, j)),
           percept_val_2_helper(mine(i, j - 1), mine(i - 1, j), mine(i, j + 1), mine(i + 1, j)),
           percept_val_2_helper(mine(i, j + 1), mine(i + 1, j), mine(i, j - 1), mine(i - 1, j)),
           percept_val_2_helper(mine(i, j - 1), mine(i + 1, j), mine(i - 1, j), mine(i, j + 1)),
           percept_val_2_helper(mine(i, j - 1), mine(i, j + 1), mine(i - 1, j), mine(i + 1, j))]
    # print(dnf)
    cnf = dnf_to_cnf(dnf, 0)
    # print(cnf)
    return cnf


def percept_val_2_helper(w, x, y, z):
    conjunction = [-w, -x, y, z]
    return conjunction


def percept_val_3(i, j):
    dnf = [[-percept(i, j, 3)],
           percept_val_3_helper(mine(i - 1, j), mine(i + 1, j), mine(i, j - 1), mine(i, j + 1)),
           percept_val_3_helper(mine(i, j + 1), mine(i - 1, j), mine(i, j - 1), mine(i + 1, j)),
           percept_val_3_helper(mine(i, j + 1), mine(i + 1, j), mine(i, j - 1), mine(i - 1, j)),
           percept_val_3_helper(mine(i, j + 1), mine(i + 1, j), mine(i - 1, j), mine(i, j - 1))]
    # print(dnf)
    cnf = dnf_to_cnf(dnf, 0)
    # print(cnf)
    return cnf


def percept_val_3_helper(w, x, y, z):
    conjunction = [w, x, y, -z]
    return conjunction


def init_knowledgebase():
    kb = Glucose3()
    # for gold in i,j: i*1000+j
    i = 2
    while i < 5:
        j = 2
        while j < 5:
            clause = [-mine(i - 1, j), -mine(i + 1, j), -mine(i, j + 1), -mine(i, j - 1), gold(i, j)]
            # condition for gold at i,j
            kb.add_clause(clause)
            cnf = dnf_to_cnf([[mine(i - 1, j), mine(i + 1, j), mine(i, j + 1), mine(i, j - 1)], [-gold(i, j)]], 0)
            for cl in cnf:
                kb.add_clause(cl)
            j += 1
        i += 1

    # for mine at i,j: i*10+j, 1<=i,j<=5
    # no mine at 0,i and i,0
    for j in range(6):
        # print(j)
        kb.add_clause([-mine(0, j + 1)])
        kb.add_clause([-mine(j + 1, 0)])
        kb.add_clause([-mine(6, j + 1)])
        kb.add_clause([-mine(j + 1, 6)])

    kb.add_clause([-mine(1, 1)])  # no mine in the 1st cell

    # for percept value k at i,j : 1<=i,j<=5 0<=k<=3 : i*100 + j*10 + k
    i = 1
    while i <= 5:
        j = 1
        while j <= 5:
            # k=0, p implies not mine for all neighbours
            cnf = percept_val_0(i, j)
            for clause in cnf:
                kb.add_clause(clause)
            # k=1
            cnf = percept_val_1(i, j)
            for clause in cnf:
                kb.add_clause(clause)
            # k=2
            cnf = percept_val_2(i, j)
            for clause in cnf:
                kb.add_clause(clause)
            # k=3
            cnf = percept_val_3(i, j)
            for clause in cnf:
                kb.add_clause(clause)

            j += 1
        i += 1
    return kb


def find_gold_location(kb):
    i = 2
    while i < 5:
        j = 2
        while j < 5:
            if ask_query(kb, gold(i, j)):
                return i, j
            j += 1
        i += 1
    return -1, -1


def plan_route(kb, curr_loc):
    # print(visited)
    # print(curr_loc)
    queue = deque([(curr_loc[0], curr_loc[1])])
    explored = set()
    explored.add((curr_loc[0], curr_loc[1]))
    parent = {(curr_loc[0], curr_loc[1]): (-1, -1)}
    target = (-1, -1)
    while len(queue) != 0:
        # print(queue)
        curr = queue.popleft()
        # if curr not in visited:
        if not ask_query(kb, visited(curr[0], curr[1])):
            target = curr
            break
        if curr[0] > 1 and ask_query(kb, -mine(curr[0] - 1, curr[1])):
            child = curr[0] - 1, curr[1]
            if child not in explored:
                queue.append(child)
                parent[child] = curr
                explored.add(child)
        if curr[0] < 5 and ask_query(kb, -mine(curr[0] + 1, curr[1])):
            child = curr[0] + 1, curr[1]
            if child not in explored:
                queue.append(child)
                parent[child] = curr
                explored.add(child)
        if curr[1] > 1 and ask_query(kb, -mine(curr[0], curr[1] - 1)):
            child = curr[0], curr[1] - 1
            if child not in explored:
                queue.append(child)
                parent[child] = curr
                explored.add(child)
        if curr[1] < 5 and ask_query(kb, -mine(curr[0], curr[1] + 1)):
            child = curr[0], curr[1] + 1
            if child not in explored:
                queue.append(child)
                parent[child] = curr
                explored.add(child)
    # print(target)
    if target == (-1, -1):
        return []

    plan = []
    while parent[target] != (-1, -1):
        if target[1] > parent[target][1]:
            plan.append("Up")
        elif target[1] < parent[target][1]:
            plan.append("Down")
        elif target[0] > parent[target][0]:
            plan.append("Right")
        else:
            plan.append("Left")
        target = parent[target]
    return plan


def main():
    st = time.time()
    ag = Agent()

    kb = init_knowledgebase()
    # for checking gold at (i,j), there shouldn't be any model satisfying -gold(i,j), should return false
    # kb.add_clause([mine(4,2)])
    # kb.add_clause([mine(3,3)])
    # kb.add_clause([mine(5,3)])
    # kb.add_clause([mine(4,4)])
    # print(kb.solve([-gold(4,3)]))
    # print(ask_query(kb,gold(4,3)))

    # checking percept correctness
    # kb.add_clause([percept(1,1,0)])
    # kb.add_clause([percept(1, 2, 1)])
    # kb.add_clause([percept(2, 1, 2)])
    # print(ask_query(kb,mine(3,1)))
    # print(kb.solve([mine(1,0)]))
    # print(ask_query(kb,mine(2,1)))

    gold_pos = (-1, -1)
    plan = []
    while gold_pos == (-1, -1):
        curr_loc = ag.FindCurrentLocation()
        if not ask_query(kb, visited(curr_loc[0], curr_loc[1])):  # if not visited
            kb.add_clause([visited(curr_loc[0], curr_loc[1])])
            kb.add_clause([percept(curr_loc[0], curr_loc[1], ag.PerceiveCurrentLocation())])
            # print(ag.PerceiveCurrentLocation())
        gold_pos = find_gold_location(kb)
        if len(plan) == 0:
            plan = plan_route(kb, curr_loc)
            # print(plan)
        if len(plan) == 0:  # no unvisited and safe cells left
            break

        action = plan.pop()
        ag.TakeAction(action)

    if gold_pos == (-1, -1):
        print("Gold could not be detected after visiting all the safe rooms.")
    else:
        print("Gold is present in room [" + str(gold_pos[0]) + "," + str(gold_pos[1]) + "]")


if __name__ == '__main__':
    main()