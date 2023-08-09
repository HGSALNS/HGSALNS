from collections import deque, defaultdict
import numpy as np
import math
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time
from random import choice


def norm1(x_coord1, y_coord1, x_coord2, y_coord2):
    """
    computing manhattan distance
    :param x_coord1: x_coord1 of node 1
    :param y_coord1: y_coord1 of node 1
    :param x_coord2: x_coord1 of node 2
    :param y_coord2: y_coord1 of node 2
    :return: manhattan distance of node1 and node 2
    """
    return abs(x_coord1 - x_coord2) + abs(y_coord1 - y_coord2)


def norm2(x_coord1, y_coord1, x_coord2, y_coord2):
    """
    computing euclidean distance
    :param x_coord1: x_coord1 of node 1
    :param y_coord1: y_coord1 of node 1
    :param x_coord2: x_coord1 of node 2
    :param y_coord2: y_coord1 of node 2
    :return: euclidean distance of node1 and node 2
    """
    return math.sqrt((x_coord1 - x_coord2) ** 2 + (y_coord1 - y_coord2) ** 2)


def read_instance(filename="../instance/results/mtmcovrpmb_instance.csv"):
    """
    read VRP instance.
    :param filename: file path.
    :return: distance cost matrix, demand list, graph of VRP node.
    """
    # initialize instances
    instance = pd.read_csv(filename)
    # save as a graph data structure
    G = nx.DiGraph()    # initialize a graph data structure
    # G.add_nodes_from([i for i in range(instance.shape[0])])
    vehicles = 0
    for i in range(instance.shape[0]):
        if instance.loc[i,'type'] == -2:
            vehicles += 1
    G.add_nodes_from([i for i in range(instance.shape[0]-vehicles)])
    nodes = []
    V = []
    for i in range(instance.shape[0]):
        if instance.loc[i,'type'] != -2:
            nodes.append([instance.loc[i,'x_coord'],instance.loc[i,'y_coord'],instance.loc[i, 'type']])
        else:
            V.append([instance.loc[i,'x_coord'],instance.loc[i,'y_coord'],instance.loc[i, 'type']])
    for i,data in enumerate(nodes):
        G.nodes[i]['x_coord'] = data[0]
        G.nodes[i]['y_coord'] = data[1]
        G.nodes[i]['type'] = data[2]
    for i in G.nodes():     # compute the Euclidean distance of node i and node j
        for j in G.nodes():
            if i != j:
                G.add_edge(i,j)
                G.edges[i,j]["distance"] = norm2(G.nodes[i]['x_coord'],G.nodes[i]['y_coord'],
                                                 G.nodes[j]['x_coord'],G.nodes[j]['y_coord'])
    D = []
    for i in G.nodes():
        temp = []
        for j in G.nodes():
            temp.append(norm2(G.nodes[i]['x_coord'],G.nodes[i]['y_coord'],
                        G.nodes[j]['x_coord'],G.nodes[j]['y_coord']))
        D.append(temp)
    D = np.array(D)
    d = []
    for i in G.nodes():
        if G.nodes[i]['type'] != -2:
            d.append(G.nodes[i]['type'])
    return D, d, G, V


def objf(sol, D, V):
    """
    A quick procedure for calclulating the quality of an solution (or a
    route). Assumes that the solution (or the route) contains all visits (incl.
    the first and the last) to the depot.
    :param sol: a route solution.
    :param D: a numpy ndarray (or equvalent) of the full 2D distance matrix.
    :return: makespan cost of an solution. [0,1,2,3,0,4,5,6,0,7,8,9,0]
    """
    ### makespan
    score = []
    for x, y, id in V:
        score.append(norm2(x, y, 0, 0))
    for route in sol:
        s = [0] + route + [0]
        distance = sum((D[s[i - 1],s[i]] for i in range(1, len(s))))
        m = min(score)
        idx = score.index(m)
        score[idx] += distance
    res = max(score)
    return res


def clarke_wright_savings(unrouted, i, D):
    """
    computing savings between node i and node j in unrouted.
    :param unrouted: unrouted customers
    :param i: customer in current route
    :param D: a numpy ndarray (or equvalent) of the full 2D distance matrix.
    :return: a savings list.
    """
    savings = [(D[i, 0] + D[0, j] - D[i, j], -D[i, j], i, j) for j in unrouted if i != j]
    savings.sort()   # ASC
    return savings


def sequential_savings_init(D, d, C, V, minimize_K=False,
                            initialize_routes_with="closest",
                            savings_callback=clarke_wright_savings):
    """
    Implementation of the Webb (1964) sequential savings algorithm /
    construciton heuristic for capaciated vehicle routing problems with
    symmetric distances.

    This is the sequential route version, which builds the solution one route
    at the time by making always the best possible merge for that active route.

    It has been implemented as separate function from the parallel version,
    because their working principle and datastrucutres differ significantly.
    :param D: a numpy ndarray (or equvalent) of the full 2D distance matrix.
    :param d: a list of demands. d[0] should be 0.0 as it is the depot.
    :param C: the capacity constraint limit for the identical vehicles.
    :param V: available vehicles list
    :param minimize_K: minimize_K sets the primary optimization objective.
    :param initialize_routes_with: rules.
    :param savings_callback: functions.
    :return: solution, assignment: routes, route assignments
    """
    # 所有路径一侧的顾客都可以进行节约合并
    N = len(D)
    ignore_negative_savings = not minimize_K    # mimimum solution/routing cost

    unrouted = set(range(1, N))    # at the beginning, only depot is routed.

    ### 1. Generate a list of seed nodes for emerging route inititialization
    # choose first customer
    if initialize_routes_with == "farthest" or initialize_routes_with == "closest":
        # build a ordered priority queue of potential route initialization nodes
        seed_customers = list(range(1, N))
        # order customers by the distance from depot to the customers and index of customers
        # 查找和仓库的距离
        seed_customers.sort(reverse=initialize_routes_with == "closest",
                            key=lambda i: (D[0][i], i))     # DSEC


    ### 2. Initialize a single emerging route at a time and then make all
    ##     feasible merges on it before moving on to the next one.

    solution = [0]    # depart from the depot
    savings = None
    emerging_route_nodes = None      # inter-node
    target = False
    # outer while depends on whether there is a unrouted customer
    # 只要有未分配的顾客都要将顾客分配到路径
    while unrouted:    # when there is a unrouted node

        ### start a new route
        if not savings:    # if saving is None represents there is new route
            # get a seed customer for a new route
            # 获取每次创建新路径的第一个点
            while True:
                # choose one unrouted customers to create a new route
                cus_idx = choice(list(range(len(seed_customers))))
                # print(f"customer {seed_customers[cus_idx]}")
                seed = seed_customers.pop(cus_idx)
                if seed in unrouted:
                    break


            # initialize a double queue
            # 获取新路径中的第一个点
            emerging_route_nodes = deque([seed])
            unrouted.remove(seed)

            # the sum of demand of vehicle
            # d is a list of demands. d[0] should be 0.0 as it is the depot
            # static capacity
            route_d = d[seed] if C else 0.0
            # dynamic capacity
            # 去程顾客
            if d[seed] < 0:
                # 记录动态的容量变化
                # 第一个顾客是去程顾客，则仓库和第一个顾客的装载量为[1, 0]
                route_dy = [-d[seed],0]
            # 回程顾客
            else:
                # 记录动态的容量变化
                # 第一个顾客是回程顾客，则仓库和第一个顾客的装载量为[0, 1]
                route_dy = [0,d[seed]]

            # compute the savings between customers in unrouted and seed
            # 计算所有未分配的节点j与节点seed合并后的节约的排序
            savings = savings_callback(unrouted, seed, D)

        ### update the new route
        # check sorted list of point pairs (savings)
        # update current route unless no savings improvement of exceed to capacities
        # one step change choose one customer for the new route
        # 只要能找到正的节约，就要一直往左侧拼接路径
        # 如果最大的节约也是负的，说明无法合并新的路径，则停止生成当前路径
        while len(savings) > 0:
            # Note: i is the one to merge with, and j is the merge_candidate
            # 获取与seed最大的节约,saving升序排列，i为主要的节点，j为要查找节约大的节点
            best_saving, _, i, j = savings.pop()

            if ignore_negative_savings:   # mimimum solution/routing cost
                cw_saving = D[i, 0] + D[0, j] - D[i, j]    # savings func
                # no improvement, choose next route
                if cw_saving < 0.0:
                    savings = []  # force route change
                    break

            # if j is routed, no action.
            if not j in unrouted:
                continue

            # check if the new route will exceed to the capacity of vehicles
            # try to add new customer but fail to add
            # BUG
            if C and route_d + d[j] > C:
                continue  # next savings
            # 检查装载量是否违反容量约束
            if d[j] < 0:
                # 之前所有的装载量+=1
                temp = [r - d[j] for r in route_dy]
                # 检查每一刻的装载量是否违反约束
                for t in temp:
                    if t > C:
                        target = True
                if target == True:
                    target = False
                    continue
            if C and route_dy[-1]+d[j] > C:
                continue

            # it is still a valid merge?
            # check where is the i in route, left or right
            # emerging_route_nodes[0]表示最左侧的节点，也是最新的节点，我们需要连接的方向
            do_left_merge = emerging_route_nodes[0] == i
            # do_right_merge = emerging_route_nodes[-1] == i and \
            #                  len(emerging_route_nodes) > 1
            # if i is the intermediate customer, no merge
            # if not (do_left_merge or do_right_merge):
            if not do_left_merge:
                continue  # next savings

            # the capacity constraint limit for the identical vehicles.
            if C:
                route_d += d[j]
                # 如果是去程顾客，要更新之前装载量的变化
                if d[j] < 0:
                    route_dy = [i - d[j] for i in route_dy]
                # 更新当前顾客的装载量变化
                route_dy.append(route_dy[-1]+d[j])


            # merge route
            # 合并路径时向左边添加新的序列
            if do_left_merge:
                emerging_route_nodes.appendleft(j)
            # if do_right_merge:
            #     emerging_route_nodes.append(j)
            # sign the node j is routed
            # 从未分配的顾客集合中移除添加到路径中的顾客
            unrouted.remove(j)

            # update the savings list
            # using new node to create savings, i -> j -> k -> h
            # 计算到与j节点相互连接的所有savings
            savings += savings_callback(unrouted, j, D)
            savings.sort()    # sort the savings

        # All savings merges tested, complete the route
        emerging_route_nodes.append(0)   # last visit is depot
        # 将新的路径拼接到整体的解上
        solution += emerging_route_nodes    # first visit is depot
        emerging_route_nodes = None     # create a new route

    print(f"solution {solution}")
    # get routes from the solution
    # solution = [0, 1, 2, 3, 0, 4, 5, 5, 0]
    solutions = []
    depots = [i for i,x in enumerate(solution) if x == 0]
    for i in range(1,len(depots)):
        # 获取[0,1,2,3,0]
        route = solution[depots[i-1]:depots[i]+1]
        # 将解翻转
        route.reverse()
        solutions.append(route)

    # solution consist of routes which need be assigned to vehicles subsequently.
    # 对路径进行分配
    ass = defaultdict(list)
    score = []
    for x, y, id in V:
        score.append(norm2(x, y, 0, 0))
    for i in range(len(solutions)):
        route = solutions[i]
        s = [0] + route + [0]
        distance = sum((D[s[i - 1], s[i]] for i in range(1, len(s))))
        m = min(score)
        idx = score.index(m)
        score[idx] += distance
        ass[idx].append(i)
    print(f"CWS solutions is {solutions}")
    return solutions, ass


def plot(res, G, ass):
    """
    plot the graph
    :param res: solution.
    :param G: graph.
    :param ass: route assignment plan
    :return: None
    """
    import matplotlib
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(V)))

    
    # depot
    X_D = []
    Y_D = []
    for id,data in G.nodes(data=True):
        if data['type'] == 0:
            X_D.append(G.nodes[id]['x_coord'])
            Y_D.append(G.nodes[id]['y_coord'])
    depot = plt.scatter(X_D, Y_D, s = 100, color ='black',marker='s')


    # linehauls
    X_L = []
    Y_L = []
    for id,data in G.nodes(data=True):
        if data['type'] == -1:
            X_L.append(G.nodes[id]['x_coord'])
            Y_L.append(G.nodes[id]['y_coord'])
    linehaul = plt.scatter(X_L, Y_L, s = 100, color='olive', marker='_', linewidth=3)
    l = len(X_L)


    # backhauls
    X_B = []
    Y_B = []
    for id,data in G.nodes(data=True):
        if data['type'] == 1:
            X_B.append(G.nodes[id]['x_coord'])
            Y_B.append(G.nodes[id]['y_coord'])
    backhaul = plt.scatter(X_B, Y_B, s = 100, color='m', marker='+', linewidth=3)
    b = len(X_B)


    # vehicle
    k = 0
    for id,data in enumerate(V):
        k += 1
        X_K = []
        Y_K = []
        X_K.append(data[0])
        Y_K.append(data[1])
        robot = plt.scatter(X_K, Y_K, s=100, color=cmap[id], marker='*', linewidth=3)


    for id,data in ass.items():
        plt.plot([0, V[id][0]], [0, V[id][1]], color=cmap[id], alpha=1.0)
        for a in data:
            r = res[a]
            for i in range(1,len(r)):
                start_x, start_y = G.nodes[r[i-1]]['x_coord'],G.nodes[r[i-1]]['y_coord']
                end_x, end_y = G.nodes[r[i]]['x_coord'], G.nodes[r[i]]['y_coord']
                plt.plot([start_x, end_x], [start_y, end_y], color=cmap[id], alpha=1.0)
    plt.legend(handles=[depot, robot, linehaul, backhaul], labels=['depot', 'robot','linehaul','backhaul'])
    plt.savefig(f"../instance/results/CW_mtmcovrpmb_k_{k}_l_{l}_b_{b}.png")
    plt.show()


if __name__ == "__main__":
    # 主程序代码开始时间
    time_start = time.time()

    # set the parameters
    ### instance 1
    D, d, G, V = read_instance()
    C = 6
    # solution = [0, 14, 5, 12, 10, 9, 11, 6, 0, 3, 7, 13, 2, 1, 8, 4, 0]
    # route 1: 0 -> 6 -> 11 -> 9 -> 10 -> 12 -> 5 -> 14 -> 0
    # route 2: 0 -> 4 -> 8 -> 1 -> 2 -> 13 -> 7 -> 3 -> 0


    solution, assignment = sequential_savings_init(D, d, C, V)
    plot(solution, G, assignment)
    print(f"solution: {solution}")
    print(f"assignment: {assignment}")
    obj = objf(solution,D, V)
    print(f"objVal: {obj}")


    # 主程序代码结束时间
    time_end = time.time()
    total_time = time_end - time_start
    hours = total_time // 3600
    minutes = (total_time - hours * 3600) // 60
    seconds = total_time % 60
    print(f"time cost: {hours} hours {minutes} minutes {seconds} seconds")