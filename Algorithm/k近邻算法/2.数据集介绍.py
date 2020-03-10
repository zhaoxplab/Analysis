import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split

# 数据集获取
# 1.1小数据集获取
iris = load_iris()
# print(iris)

# 1.2 大数据集获取
# news = fetch_20newsgroups()
# print(news)

# 2.数据集属性描述
# print('数据集特征值是:\n', iris.data)
# print('数据集目标是:\n', iris['target'])
# print('数据集特征值的名字是:\n', iris.feature_names)
# print('数据集目标值的名字是:\n', iris.target_names)
# print('数据集的描述是:\n', iris.DESCR)


# 3.数据集的可视化
# 把数据转换成dataframe的格式
iris_d = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
iris_d['target'] = iris.target
# print(iris_d)


def iris_plot(data, col1, col2):
    sns.lmplot(x=col1, y=col2, data=data, hue='target', fit_reg=False)
    plt.title('鸢尾花数据显示')
    plt.show()


# iris_plot(iris_d, 'sepal_width', 'petal_length')
# iris_plot(iris_d, 'sepal_length', 'petal_width')


# 数据集的划分
# 对鸢尾花数据集进行分割
# 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
print('训练集的特征值是\n', x_train)
print('训练集的目标值是\n', y_train)
print('测试集的特征值是\n', x_test)
print('测试集的目标值是\n', y_test)

print('训练集目标值的形状是\n', y_train.shape)
print('测试集目标值的形状是\n', y_test.shape)

x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, random_state=6)
x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, random_state=6)
# print("如果随机数种子不一致：\n", x_train == x_train1)
# print("如果随机数种子一致：\n", x_train1 == x_train2)