### 仓库代码说明

## constructive_heuristic
CWS.py文件是构造型启发式算法的执行文件。

## formulation
integer_formulation.py文件是数学规划模型的执行文件。

## instance
generate.py文件是仿真算例生成的执行文件，dataset文件夹是论文所用的测试算例的数据集。其中，仿真算例采用顾客均匀随机分布进行生成。列名包括id,x_coord,y_coord,type。id表示节点的唯一识别符；x_coord和y_coord表示其对应坐标位置；type表示节点类型，type=0表示工作台，type=-1表示去程顾客，type=1表示回程顾客，type=-2表示车辆。
