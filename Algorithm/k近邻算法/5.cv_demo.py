from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# 1.获取数据
iris = load_iris()

# 2.数据基本处理
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

# 3.特征工程 - 特征预处理
# 实例化一个转换器类
transfer = StandardScaler()
# 传入训练集的特征值，使用fit_transform进行训练集特征值的转换
x_train = transfer.fit_transform(x_train)
# 传入测试集的特征值，
x_test = transfer.transform(x_test)

# 4.机器学习-KNN
# 4.1 实例化一个估计器
estimator = KNeighborsClassifier()
# 4.1模型调优-交叉验证
param_grid = {'n_neighbors': [1, 3, 5, 7]}
# 将估计器传入进行交叉验证
estimator = GridSearchCV(estimator, param_grid=param_grid, cv=5)
# 4.2 模型训练
estimator.fit(x_train, y_train)

# 5.模型评估
# 5.1 预测值结果输出
y_pre = estimator.predict(x_test)
print('预测值是：\n', y_pre)
# 对比真实值和预测值
print('预测值和真实值的对比是：\n', y_pre == y_test)

# 5.2 准确率计算
score = estimator.score(x_test, y_test)
print('准确率为：\n', score)

# 然后进行评估查看最终选择的结果和交叉验证的结果
print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
print("最好的参数模型：\n", estimator.best_estimator_)
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)
