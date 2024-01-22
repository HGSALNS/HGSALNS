import pandas as pd
import random


random.seed(20)
# 本研究采用randrange均匀随机整数生成以及random均匀随机小数生成并相加的方式生成均匀随机分布。
# 我们也可以采用random.uniform的方式来生成均匀随机分布的仿真算例，仿真算例的生成方式并不影响论文的实验结论。

def generate(vehicle=3,linehaul=10,backhaul=10,size=40):
    """
    type=0表示depot，type=-1表示linehaul，type=1表示backhaul，type=-2表示vehicle
    """
    visit = set()
    id = 0
    # 创建depot位置，id=0，x_coord=0, y_coord=0, type = 0
    instance = pd.DataFrame(columns=['id','x_coord','y_coord','type'])
    instance.loc[len(instance.index)] = [id,0,0,0]
    visit.add((0,0))
    id += 1
    # 添加linehaul的算例
    linehaul_set = []
    while len(linehaul_set) != linehaul:
        x_coord = random.randrange(-size//2,size//2) + random.random()
        y_coord = random.randrange(-size//2,size//2) + random.random()
        if (x_coord,y_coord) not in visit:
            instance.loc[len(instance.index)] = [id,x_coord,y_coord,-1]
            linehaul_set.append([id])
            visit.add((x_coord,y_coord))
            id += 1
    # 添加backhaul的算例
    backhaul_set = []
    while len(backhaul_set) != backhaul:
        x_coord = random.randrange(-size//2,size//2) + random.random()
        y_coord = random.randrange(-size//2,size//2) + random.random()
        if (x_coord,y_coord) not in visit:
            instance.loc[len(instance.index)] = [id,x_coord,y_coord,1]
            backhaul_set.append([id])
            visit.add((x_coord,y_coord))
            id += 1
    # 添加vehicle的算例
    vehicle_set = []
    while len(vehicle_set) != vehicle:
        x_coord = random.randrange(-size//2,size//2) + random.random()
        y_coord = random.randrange(-size//2,size//2) + random.random()
        if (x_coord,y_coord) not in visit:
            instance.loc[len(instance.index)] = [id,x_coord,y_coord,-2]
            vehicle_set.append([id])
            visit.add((x_coord,y_coord))
            id += 1
    print(instance)
    instance.to_csv("../instance/results/mtmcovrpmb_instance.csv",index=False)
    return


generate(6, 23, 19, 30)
