from sklearn.datasets import load_boston  # 获取数据
import matplotlib.pyplot as plt
import pandas as pd


# 获取数据集
boston = load_boston()
# print(boston)

# 2.数据集属性描述
print('数据集特征值是:\n', boston.data)
print('数据集目标是:\n', boston['target'])
print('数据集特征值的名字是:\n', boston.feature_names)
print('数据集目标值的名字是:\n', boston.filename)
print('数据集的描述是:\n', boston.DESCR)
