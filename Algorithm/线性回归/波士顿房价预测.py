"""
# 1.获取数据
# 2.数据基本处理
# 2.1 分割数据
# 3.特征工程-标准化
# 4.机器学习-线性回归
# 5.模型评估
"""
from sklearn.datasets import load_boston  # 获取数据
from sklearn.model_selection import train_test_split  # 数据处理，分割数据
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error


def linear_model1():
    """
    线性回归：正规方程
    :return:
    """
    # 1.获取数据
    boston = load_boston()
    # print(boston)
    # 2.数据基本处理
    # 2.1 分割数据
    # 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 3.特征工程-标准化
    teransfer = StandardScaler()
    x_train = teransfer.fit_transform(x_train)
    x_test = teransfer.fit_transform(x_test)
    # 4.机器学习-线性回归
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    print('这个模型的偏置是:', estimator.intercept_)
    print('这个模型的系数是:', estimator.coef_)
    # 5.模型评估
    # 5.1 预测值
    y_pre = estimator.predict(x_test)
    print('预测值是:', y_pre)
    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pre)
    print('均方误差:', ret)
    pass


def linear_model2():
    """
    线性回归：梯度下降
    :return:
    """
    # 1.获取数据
    boston = load_boston()
    print(boston)
    # 2.数据基本处理
    # 2.1 分割数据
    # 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 3.特征工程-标准化
    teransfer = StandardScaler()
    x_train = teransfer.fit_transform(x_train)
    x_test = teransfer.fit_transform(x_test)
    # 4.机器学习-线性回归
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)

    print('这个模型的偏置是:', estimator.intercept_)
    print('这个模型的系数是:', estimator.coef_)
    # 5.模型评估
    # 5.1 预测值
    y_pre = estimator.predict(x_test)
    print('预测值是:', y_pre)
    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pre)
    print('均方误差:', ret)
    pass


# 正规方程
# linear_model1()
# 梯度下降
linear_model2()


