# 训练svm模型
# 导入包
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

# 解析数据
train = []
labels = []
train_open_txt = open('train_open.txt', 'r')
# zzz = train_open_txt.readlines()
print('Reading train_open.txt...')
line_ctr = 0
for txt_str in train_open_txt.readlines():
    temp = []
    # print(txt_str)
    datas = txt_str.strip()
    datas = datas.replace('[', '')
    datas = datas.replace(']', '')
    datas = datas.split(',')
    print(datas)
    for data in datas:
        # print(data)
        data = float(data)
        temp.append(data)
    # print(temp)
    train.append(temp)
    labels.append(0)

print('Reading train_close.txt...')
line_ctr = 0
temp = []
train_close_txt = open('train_close.txt', 'r')
for txt_str in train_close_txt.readlines():
    temp = []
    # print(txt_str)
    datas = txt_str.strip()
    datas = datas.replace('[', '')
    datas = datas.replace(']', '')
    datas = datas.split(',')
    print(datas)
    for data in datas:
        # print(data)
        data = float(data)
        temp.append(data)
    # print(temp)
    train.append(temp)
    labels.append(1)

for i in range(len(labels)):
    print("{0} --> {1}".format(train[i], labels[i]))

train_close_txt.close()
train_open_txt.close()

# print(train)
# print(labels)

# 训练并保存模型
# 功能：分隔超平面
# 参数：C:  float参数 默认值为1.0
#           错误项的惩罚系数。C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，
#           但是泛化能力降低，也就是对测试数据的分类准确率降低。
#           相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。
#           对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。
#       kernel: str参数 默认为‘rbf’
#               算法中采用的核函数类型，可选参数有：
#               ‘linear’:线性核函数
#               ‘poly’：多项式核函数
#               ‘rbf’：径像核函数/高斯核
#               ‘sigmod’:sigmod核函数
#               ‘precomputed’:核矩阵
#       gamma: float参数 默认为auto
#               gamma值越小，分类界面越连续；gamma值越大，分类界面越“散
#               核函数系数，只对‘rbf’,‘poly’,‘sigmod’有效。
#               如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features.
#       decision_function_shape：'ovr'：表示one v rest，即一个类别与其他类别划分，多分类；
#                                'ovo'：表示one v one，即一个类别与另一个类别划分，二分类
# 返回值：svm模型
clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
clf.fit(train, labels)
joblib.dump(clf, "ear_svm.m")

# 测试准确率
print('predicting [[0.34, 0.34, 0.31]]')
res = clf.predict([[0.34, 0.34, 0.31]])
print(res)

print('predicting [[0.19, 0.18, 0.18]]')
res = clf.predict([[0.19, 0.18, 0.18]])
print(res)


