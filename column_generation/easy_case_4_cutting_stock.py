"""
@Project  ：extract_algorithm 
@File     ：easy_case_4_cutting_stock.py
@IDE      : PyCharm
@Author   ：Pan YX
@Time     ：2024/7/25 21:39
@Describ  ：
"""

from gurobipy import *

TypesDemand = [3, 7, 9, 16]  # 需求长度
QuantityDemand = [25, 30, 14, 8]  # 需求的量
LengthUsable = 20  # 钢管长度
rmp_count = 1  # 主问题迭代计数
sub_cont = 1  # 子问题迭代计数

MainProbRelax = Model()  # 松弛后的列生成主问题 RMP
SubProb = Model()  # 子问题

# 构造主问题模型，选择的初始切割方案 每根钢管只切一种长度
# 添加变量
Zp = MainProbRelax.addVars(len(TypesDemand), obj=1.0, vtype=GRB.CONTINUOUS, name='z')  # 设置主问题变量和目标函数中的成本系数.
# 添加约束
ColumnIndex = MainProbRelax.addConstrs(
    quicksum(Zp[p] * (LengthUsable // TypesDemand[i]) for p in range(len(TypesDemand)) if p == i)
    >=
    QuantityDemand[i] for i in range(len(TypesDemand))
)
MainProbRelax.setParam('Outputflag', 0)
MainProbRelax.optimize()  # 求解RMP,得到对应的对偶变量值,构建PP定价子问题.
MainProbRelax.write('model/RMP{}.lp'.format(rmp_count))
rmp_count += 1

# 构造子问题模型 PP
# 获得对偶值
Dualsolution = MainProbRelax.getAttr(GRB.Attr.Pi, MainProbRelax.getConstrs())
# 添加变量
Ci = SubProb.addVars(len(TypesDemand), obj=Dualsolution, vtype=GRB.INTEGER, name='c')  # c对应具体的一个切割方案.
# 添加约束
SubProb.addConstr(quicksum(Ci[i] * TypesDemand[i] for i in range(len(TypesDemand))) <= LengthUsable)  # 保证切割方案满足可行性.
# SubProb.setAttr(GRB.Attr.ModelSense, -1)
SubProb.setObjective(1 - quicksum(Ci[i] * Dualsolution[i] for i in range(len(Dualsolution))),
                     GRB.MINIMIZE)  # 设置子问题目标函数.
SubProb.setParam('Outputflag', 0)
SubProb.optimize()  # 求解
SubProb.write('model/subprob{}.lp'.format(sub_cont))
sub_cont += 1

'''判断Reduced Cost是否小于零
1. 检验数满足,可以增加新列,RMP增加一列,目标函数增加一个变量.
2. 检验数不满足,退出循环.
第一次解为: [2,2,0,0] obj = -0.33333333333333326 满足条件,修改RMP模型和PP目标函数成本系数.
第二次解为: [0,0,2,0] obj = 1.0 不满足条件,推出循环.
'''
while SubProb.objval < 0:  # 检验数小于0,说明有变量可以进基.
    # 获取变量取值
    columnCoeff = SubProb.getAttr("X", SubProb.getVars())
    column = Column(columnCoeff, MainProbRelax.getConstrs())  # 生成RMP中的一列.
    print(f'加入的列系数为:{column._coeffs}')
    # 添加变量
    MainProbRelax.addVar(obj=1.0, vtype=GRB.CONTINUOUS, name="CG", column=column)
    MainProbRelax.optimize()  # 求解修改后的RMP
    MainProbRelax.write('model/RMP{}.lp'.format(rmp_count))
    rmp_count += 1
    # 修改子问题目标函数系数
    for i in range(len(TypesDemand)):
        Ci[i].obj = ColumnIndex[i].pi
    SubProb.optimize()
    sub_pro_obj_val = SubProb.objval
    if sub_pro_obj_val < 0:
        print(f'子问题目标函数为{sub_pro_obj_val},满足迭代条件.')
    else:
        print(f'子问题目标函数为{sub_pro_obj_val},不满足迭代条件.')
    SubProb.write('model/subprob{}.lp'.format(sub_cont))
    sub_cont += 1

# 将CG后的模型转为整数，并求解
for v in MainProbRelax.getVars():
    v.setAttr("VType", GRB.INTEGER)
MainProbRelax.optimize()
for v in MainProbRelax.getVars():
    if v.X != 0.0:
        print('%s %g' % (v.VarName, v.X))