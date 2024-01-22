from itertools import combinations

import pandas as pd
import numpy as np
import math
from gurobipy import Model,GRB,tuplelist,tupledict,quicksum
import networkx as nx
import time
import matplotlib.pyplot as plt
import datetime


# def norm1(x_coord1, y_coord1, x_coord2, y_coord2):
#     return abs(x_coord1 - x_coord2) + abs(y_coord1 - y_coord2)

def norm2(x_coord1, y_coord1, x_coord2, y_coord2):
    return math.sqrt((x_coord1 - x_coord2) ** 2 + (y_coord1 - y_coord2) ** 2)

# instance.csv
def read_instance(filename="../instance/results/mtmcovrpmb_instance.csv"):
    # 加载算例
    instance = pd.read_csv(filename)
    # 将算例结果保存为图模型
    G = nx.DiGraph()    # 初始化图模型
    # 添加所有任务节点信息
    G.add_nodes_from([i for i in range(instance.shape[0])])
    for i in range(instance.shape[0]):
        G.nodes[i]['x_coord'] = instance.loc[i,'x_coord']
        G.nodes[i]['y_coord'] = instance.loc[i,'y_coord']
        G.nodes[i]['type'] = instance.loc[i, 'type']
    for i in G.nodes():     # 计算任意两个节点之间的欧式距离
        for j in G.nodes():
            if i != j:
                G.add_edge(i,j)
                G.edges[i,j]["distance"] = norm2(G.nodes[i]['x_coord'],G.nodes[i]['y_coord'],
                                                 G.nodes[j]['x_coord'],G.nodes[j]['y_coord'])
    return G


def solve_model(Q=6,T_max=3,func="total_cost",output_para=False,output_obj=True):
    # 获取算例中的图模型
    G = read_instance()
    # 所有点元组列表
    N = tuplelist([k for k in G.nodes()])

    N_add_M = tuplelist([k for k in G.nodes()] + ["m"])

    # linehaul
    L = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] == -1])
    # backhaul
    B = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] == 1])
    # depot
    D = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] == 0])
    # vehicle
    K = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] == -2])

    # N_sub_L
    N_sub_L = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] != -1])
    # N_sub_B
    N_sub_B = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] != 1])
    # N_sub_D
    N_sub_D = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] != 0])
    # N_sub_K
    N_sub_K = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] != -2])
    # N_sub_K_add_M
    N_sub_K_add_M = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] != -2] + ["m"])

    # B_add_L
    B_add_L = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] == 1 or G.nodes[k]['type'] == -1])
    # B_add_D
    B_add_D = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] == 1 or G.nodes[k]['type'] == 0])
    # B_add_K
    B_add_K = tuplelist([k for k in G.nodes() if G.nodes[k]['type'] == 1 or G.nodes[k]['type'] == -2])

    # trip
    T = tuplelist([k for k in range(T_max)])
    # trip_sub_0
    T_sub_0 = tuplelist([k for k in range(1,T_max)])
    # trip_sub_max
    T_sub_max = tuplelist([k for k in range(T_max-1)])
    # 成本字典
    dist = {(i,j):norm2(G.nodes[i]['x_coord'],G.nodes[i]['y_coord'],
                        G.nodes[j]['x_coord'],G.nodes[j]['y_coord'])
            for i in G.nodes() for j in G.nodes()}


    # 实例化模型
    model = Model("mtmcovrpmb")
    model.setParam('OutputFlag', 0)   # 不输出日志消息

    # 添加变量并设置目标函数
    X = model.addVars(N, N_add_M, K, T, vtype=GRB.BINARY, name='X')
    U = model.addVars(N, K, T, vtype=GRB.INTEGER, name='U')

    if func == "total_cost":
        # 设置sum目标函数
        z1 = 0
        for k in K:
            for t in T:
                z1 += quicksum( X[i,j,k,t] * dist[(i,j)] for i in N for j in N if i != j)
        model.setObjective(z1, GRB.MINIMIZE)
    elif func == "distance_span":
        # 设置makespan目标函数
        C = model.addVar(lb = 0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'continuous')
        model.addConstrs( quicksum( X[i,j,k,t] * dist[(i,j)] for i in N for j in N for t in T) <= C for k in K)
        model.setObjective(C, GRB.MINIMIZE)

    ### 添加约束
    ## vertex约束
    # vehicle约束
    # (2) in-degree constraint
    model.addConstrs( X[i,j,k,t] == 0 for i in N for j in K for k in K for t in T)
    # (3) out-degree constraint first trip, start from vehicle
    model.addConstrs( quicksum( X[k,j,k,0] for j in B_add_D) <= 1 for k in K)
    # (4) out-degree constraint first trip, only can start from vehicle respond point
    model.addConstrs(quicksum(X[i, j, k, 0] for i in K for j in N if i != k) == 0 for k in K)
    # (5) out-degree constraint first trip, doesn't to linehaul
    # 只要保证linehaul顾客在第一次行程不会被访问，则不会出现linehaul到bachaul或backhaul到linehaul情况
    model.addConstrs(X[i, j, k, 0] == 0 for j in L for i in N for k in K)
    # (6) out-degree constraint last trip, doesn't start vehicle unless trip 0
    model.addConstrs( X[i,j,k,t] == 0 for i in K for j in N for t in T_sub_0 for k in K)
    # depot约束
    # (7) in-degree constraint
    model.addConstrs( quicksum( X[i,0,k,t] for i in N) <= 1 for k in K for t in T)
    # (8) out-degree constraint trip 0
    model.addConstrs( quicksum( X[0,j,k,0] for j in N) == 0 for k in K)
    # (9) out-degree constraint trip-
    model.addConstrs( quicksum( X[0,j,k,t] for j in N) <= 1 for k in K for t in T_sub_0)
    # (10) balance constraint
    model.addConstrs( quicksum( X[i,0,k,t] for i in N) >= quicksum( X[0,j,k,t+1] for j in N) for k in K for t in T_sub_max)
    # linehaul约束
    # (11) in-degree constraint
    model.addConstrs( quicksum( X[j,i,k,t] for k in K for t in T for j in N if i != j) == 1 for i in B_add_L)
    # (12) backhaul balance constraint first trip
    model.addConstrs( quicksum( X[i,j,k,0] for i in B_add_K) == quicksum( X[j,i,k,0] for i in B_add_D) for j in B for k in K)
    # (13) backhaul balance constraint last trip
    model.addConstrs( quicksum( X[i,j,k,t] for i in N) == quicksum( X[j,i,k,t] for i in N) for j in B for t in T_sub_0 for k in K)
    # (14) linehaul balance constraint last trip
    model.addConstrs( quicksum( X[i,j,k,t] for i in N) == quicksum( X[j,i,k,t] for i in N_add_M) for j in L for t in T for k in K)
    # (15) back constraint
    for t in T:
        for k in K:
            for i in B:
                for j in N_sub_K:
                    if i != j:
                        model.addConstr((X[i, j, k, t] == 1) >> (quicksum(X[i, 0, k, t] for i in N_sub_D) == 1), name='indicator')
    # (16) dummy constraint
    model.addConstrs( quicksum( X[i,"m",k,t] for t in T for i in N) <= 1 for k in K)
    ## trip约束
    # capacity约束
    # (17) customers capacity constraint
    for i in N:
        for j in B_add_L:   # N修改为B_add_L,最后一个到达点depot，会一下子完成所有backhaul任务
            for t in T:
                for k in K:
                    if i != j:
                        model.addConstr((X[i,j,k,t] == 1) >> (U[i,k,t] + G.nodes[j]['type'] == U[j,k,t]), name="indicator")
    # (18) capacity satisfy constraint
    model.addConstrs( U[i,k,t] <= Q for i in N for t in T for k in K)
    # (19) depot capacity constraint
    model.addConstrs( U[0,k,t] == quicksum( X[i,j,k,t] * (-G.nodes[i]['type']) for i in L for j in N_sub_K_add_M if i != j) for t in T for k in K)
    # (20) vehicle capacity constraint
    model.addConstrs( U[k,k,0] == 0 for k in K)
    ## MTZ约束
    # (21) MTZ约束
    e = model.addVars(N, vtype=GRB.INTEGER, name='e')   # i in B_add_L 和 i in N_sub_D 结果不一样
    model.addConstrs(e[i] - e[j] + 2 * Q * quicksum(X[i,j,k,t] for k in K for t in T) <= 2 * Q - 1 for i in B_add_L for j in B_add_L if i != j)
    ## 决策变量
    model.addConstrs( U[i,k,t] >= 0 for i in N for k in K for t in T)

    model.Params.TimeLimit = 72000  # 设置求解时间上限
    model.optimize()

    # 获取求解模型的解
    res = []
    if model.status == GRB.Status.OPTIMAL or model.status == GRB.Status.TIME_LIMIT:
        # 保存结果
        for i in N:
            for j in N:
                for k in K:
                    for t in T:
                        if i != j and X[i,j,k,t].x > 0.5:
                            res.append([i,j,k])
        # 打印结果
        if output_para:
            for v in model.getVars():
                if v.x > 0.5 and 'X' in v.varName:
                    print('参数', v.varName, '=', v.x)
                if "U" in v.varName:
                    print(f"参数 {v.varName} = {v.x}")
        if output_obj:
            print(f"最优值 {model.objVal}")
    return res,G



def plot(res,G):
    """
    绘制mtvrpmb解的图像
    :param res: 使用嵌套列表存储解[[i,j,k]]
    :param G: 图结构
    :return: None
    """
    import matplotlib
    vehicles_number = len([i for i in G.nodes() if G.nodes[i]['type'] == -2])
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, vehicles_number))

    cus = len([i for i in G.nodes() if G.nodes[i]['type'] != -2])

    # 绘制城市坐标
    X_D = []
    Y_D = []
    for id,data in G.nodes(data=True):
        if data['type'] == 0:
            X_D.append(G.nodes[id]['x_coord'])
            Y_D.append(G.nodes[id]['y_coord'])
    depot = plt.scatter(X_D, Y_D, s = 100, color ='black',marker='s')
    X_L = []
    Y_L = []
    for id,data in G.nodes(data=True):
        if data['type'] == -1:
            X_L.append(G.nodes[id]['x_coord'])
            Y_L.append(G.nodes[id]['y_coord'])
    linehaul = plt.scatter(X_L, Y_L, s = 100, color='olive', marker='_', linewidth=3)
    l = len(X_L)
    X_B = []
    Y_B = []
    for id,data in G.nodes(data=True):
        if data['type'] == 1:
            X_B.append(G.nodes[id]['x_coord'])
            Y_B.append(G.nodes[id]['y_coord'])
    backhaul = plt.scatter(X_B, Y_B, s = 100, color='m', marker='+', linewidth=3)
    b = len(X_B)
    k=0
    for id,data in G.nodes(data=True):
        X_K = []
        Y_K = []
        if data['type'] == -2:
            k+=1
            X_K.append(G.nodes[id]['x_coord'])
            Y_K.append(G.nodes[id]['y_coord'])
            robot = plt.scatter(X_K, Y_K, s=100, color=cmap[id-cus], marker='*', linewidth=3)
    for i,j,k in res:
        start_x, start_y = G.nodes[i]['x_coord'],G.nodes[i]['y_coord']
        end_x, end_y = G.nodes[j]['x_coord'], G.nodes[j]['y_coord']
        plt.plot([start_x, end_x], [start_y, end_y], color=cmap[k-cus], alpha=1.0)
    plt.legend(handles=[depot, robot, linehaul, backhaul], labels=['depot', 'robot','linehaul','backhaul'])
    plt.savefig(f"../instance/results/formulation_mtmcovrpmb_k_{k}_l_{l}_b_{b}.png")
    plt.show()



if __name__ == "__main__":
    # 主程序代码开始时间
    time_start = time.time()

    res, G = solve_model(6,3,"distance_span",True)
    plot(res, G)

    # 主程序代码结束时间
    time_end = time.time()
    total_time = time_end - time_start
    hours = total_time // 3600
    minutes = (total_time - hours * 3600) // 60
    seconds = total_time % 60
    print(f"time cost: {hours} hours {minutes} minutes {seconds} seconds")
