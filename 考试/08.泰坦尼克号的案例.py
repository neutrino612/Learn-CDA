http://localhost:8889/notebooks/%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7%E6%A1%88%E4%BE%8B.ipynb

本地的py文件:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("E:/数据分析/SKlearn全部章节数据及代码/01 决策树数据及代码/Taitanic data")
data.info 

# 以表的形式展现出来，前多少行
data.head()   
# 筛选特征
# 删除无用的特征,inplace  为True,用删除后的表覆盖原表，false   则不会覆盖原表，需要用data承接一下,axis=1对列操作删除
data.drop(['Cabin','Name','Ticket'],inplace=True,axis=1)
data=data.drop(['Cabin','Name','Ticket'],inplace=False)
# 处理缺失值,fillna   填充；数值型的可以用均值填充
data["Age"]=data["Age"].fillna(data["Age"].mean())  
# 删除有缺失值的行数据
data=data.dropna(axis=0)

# 将文字型的特征转化成数字
# 第一步取出文字型特征的压缩结果，即去除重复值,tolist将数组转化成列表格式
labels=data["Embarked"].unique().tolist()
# apply 执行一个匿名函数  lambda x；这个函数的意义是  将这个列表的项转化成其索引
data["Embarked"]=data["Embarked"].apply(lambda x:labels.index(x))
# 直接进行一个转换，判断性别为男人的数据，得到一个true和false的一个布尔值
# loc是按照行列的取法取出特征，取得是文字型的特征；数值型的特征用iloc
data.loc[:,"Sex"]=(data["Sex"]=="male")
# 将这个数据中的true和false进行类型的转化
(data["Sex"]=="male").astype("int") 
# 此处数据预处理已经完成，下面进行特征和标签的提取

# 特征的提取，X为特征矩阵
x=data.iloc[:,data.columns !="Survived"]
# 取出标签的矩阵
y=data.iloc[:,data.columns == "Survived"]
# 进行训练集和测试集的划分
from sklearn.model_selection  import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.3)
# 剪切完后的训练集，索引是乱的，需要纠正索引。将其索引重新排序0-621之间排序,因为Xtrain.shape[0]=622；但是有四个数据集，都要进行一遍这样的操作Xtrain.index=range(Xtrain.shape[0])
for i in [Xtrain,Xtest,Ytrain,Ytest]:
    i.index=range(i.shape[0]) 











