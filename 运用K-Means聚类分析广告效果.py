# # -*- encoding = utf-8 -*-
# '''
#
# 本案例通过各种广告渠道90天内额日均UV，平均注册率，平均探索率，访问深度、
# 订单转化率。投放时间，素材类型，广告类型、合作方式。广告尺寸和广告买点等特征，将渠道分类
# 找出每类渠道的重点特征，为接下来的业务讨论和数据集分析提供支持
#
# '''

# import  numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
# from sklearn.metrics import silhouette_score
# from sklearn.cluster import KMeans
# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# raw_data=pd.read_csv("ad_performance.csv")
# # print(raw_data.head())
# # print(raw_data.info())#查看数据类型分布
# # #打印原始数据基本描述性信息
# # print(raw_data.describe().round(2).T)
# #查看缺失值
# #查看每一列书否具有缺失值
# # na_cols=raw_data.isnull().any(axis=0)
# # print(na_cols)
# # #查看缺失值总记录
# # print(raw_data.isnull().sum().sort_values(ascending=False))
# # print(raw_data.corr().round(2).T)
#
# import seaborn as sns
# corr=raw_data.corr().round(2)
# sns.heatmap(corr,cmap="Reds",annot=True)
# # plt.show()
# """
# 访问深度和平均停留时间相关性比较高，相关性比较高说明两个变量在建立模型是
# 作用一样或者效果一样，可以考虑组合或者删除其一
# """
#
# #删除平均停留时间
# raw_data2=raw_data.drop(["平均停留时间"],axis=1)
# # print(raw_data2.info())
# # print(raw_data2.head())
# cols=["素材类型","广告类型","合作方式","广告尺寸","广告卖点"]
# # for x in cols:
# #     data=raw_data2[x].unique()
# #     print("变量[{0}]的取值有：\n{1}".format(x,data))
# #     print("*****"*20)
# cols = ['素材类型','广告类型','合作方式','广告尺寸','广告卖点']
# model_ohe=OneHotEncoder(sparse=False)#建立OneHGotEncode对象
# ohe_matrix=model_ohe.fit_transform(raw_data2[cols]).astype("int")
# # print(ohe_amtris[:2])
# ohe_matrix1=pd.get_dummies(raw_data2[cols])
# # print(ohe_matrix1.head(2))
# scale_matrix=raw_data2.iloc[:,2:7]
# # print(scale_matrix)
# model_scale=MinMaxScaler()
# data_scaled=model_scale.fit_transform(scale_matrix)
# # print(data_scaled)
# #建立模型
# X = np.hstack((data_scaled, ohe_matrix))
# #通过平均轮廓系数检测得到最佳Kmean聚类模型
# sore_list=list()#盈利啊存储每个K下模型的平均轮廓数
# silhouette_int=-1#初始化平均轮廓系数阀值
# for n_cluster in range(2,8):
#     #建立模型对象
#     model_kmeans=KMeans()
#     #训练聚类模型
#     label_tem=model_kmeans.fit_predict(X)
#     #得到每一个K的平均轮廓系数
#     silhouette_tem=silhouette_score(X,label_tem)
#     if silhouette_tem>silhouette_int:#如果平均轮廓系数更高
#         best_k=n_cluster #保存k
#         silhouette_int=silhouette_tem#保存平均轮廓得分
#         besk_mean=model_kmeans
#         cluster_labels_k=label_tem
#     sore_list.append([n_cluster,silhouette_tem])
# # print('{:*^60}'.format('K值对应的轮廓系数:'))
# # print(np.array(sore_list))  # 打印输出所有K下的详细得分
# # print('最优的K值是:{0} \n对应的轮廓系数是:{1}'.format(best_k, silhouette_int))
# #
#
#
# cluster_labels=pd.DataFrame(cluster_labels_k,columns=["clusters"])
# merge_data=pd.concat((raw_data2,cluster_labels),axis=1)
# # print(merge_data)
#
# clustering_count=pd.DataFrame(merge_data["渠道代号"].
#             groupby(merge_data['clusters']).count()).T.rename({"渠道代号": 'counts'})
# clustering_ratio = (clustering_count / len(merge_data)).round(2).\
#     rename({'counts': 'percentage'})  # 计算每个聚类类别的样本量占比
# # print(clustering_count)
# # print(clustering_ratio)
#
# # print(best_k)
# cluster_feature=[]
# for line in range(best_k):
#     label_data = merge_data[merge_data['clusters'] == line]
#
#     part1_data=label_data.iloc[:,1:7]
#     part1_desc=part1_data.describe().round(3)
#     merge_data1=part1_desc.iloc[2,:]
#
#     part2_data=label_data.iloc[:,7:-1]
#     part2_desc=part1_data.describe(include="all")
#     merge_data2=part1_desc.iloc[2,:]
#     merge_line=pd.concat((merge_data1,merge_data2),axis=1)
# cluster_pd = pd.DataFrame(cluster_feature).T  # 将列表转化为矩阵
# # print('{:*^60}'.format('每个类别主要的特征:'))
# all_cluster_set = pd.concat((clustering_count, clustering_ratio, cluster_pd),axis=0)  # 将每个聚类类别的所有信息合并
# # print(all_cluster_set)
#
# num_sets = cluster_pd.iloc[:6, :].T.astype(np.float64)  # 获取要展示的数据
# num_sets_max_min = model_scale.fit_transform(num_sets)  # 获得标准化后的数据
# print(num_sets)
# print('-'*20)
# print(num_sets_max_min)
#
# import pandas as pd
# import numpy as np
#
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
# from sklearn.metrics import silhouette_score # 导入轮廓系数指标
# from sklearn.cluster import KMeans # KMeans模块
#
# ## 设置属性防止中文乱码
# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# raw_data = pd.read_csv('ad_performance.csv')
# # print(raw_data.head())
# # # 查看基本状态
# # print(raw_data.head(2) ) # 打印输出前2条数据
# # print(raw_data.info())# 打印数据类型分布
# # print(raw_data.describe().round(2).T)
# # # 缺失值审查
# # na_cols = raw_data.isnull().any(axis=0)  # 查看每一列是否具有缺失值
# # print(na_cols)
# # print(raw_data.isnull().sum().sort_values(ascending=False))# 查看具有缺失值的行总记录数
# # print(raw_data.corr().round(2).T)
# # 相关性可视化展示
# import seaborn as sns
# corr = raw_data.corr().round(2)
# sns.heatmap(corr,cmap='Reds',annot = True)
# plt.show()
# # 1 删除平均平均停留时间列
# raw_data2 = raw_data.drop(['平均停留时间'],axis=1)
# # 类别变量取值
# cols=["素材类型","广告类型","合作方式","广告尺寸","广告卖点"]
# for x in cols:
#     data=raw_data2[x].unique()
#     # print("变量【{0}】的取值有：\n{1}".format(x,data))
#     # print("-·"*20)
# # 字符串分类独热编码处理
# cols = ['素材类型','广告类型','合作方式','广告尺寸','广告卖点']
# model_ohe = OneHotEncoder(sparse=False)  # 建立OneHotEncode对象
# ohe_matrix = model_ohe.fit_transform(raw_data2[cols])  # 直接转换
# # print(ohe_matrix[:2])
# # 用pandas的方法
# ohe_matrix1=pd.get_dummies(raw_data2[cols])
# # print(ohe_matrix1.head(5))
# # 数据标准化
# sacle_matrix = raw_data2.iloc[:, 2:7]  # 获得要转换的矩阵
#
# model_scaler = MinMaxScaler()  # 建立MinMaxScaler模型对象
# data_scaled = model_scaler.fit_transform(sacle_matrix)  # MinMaxScaler标准化处理
# # print(data_scaled.round(2))
# X = np.hstack((data_scaled, ohe_matrix))
# # 通过平均轮廓系数检验得到最佳KMeans聚类模型
# score_list = list()  # 用来存储每个K下模型的平局轮廓系数
# silhouette_int = -1  # 初始化的平均轮廓系数阀值
# for n_clusters in range(2, 9):  # 遍历从2到5几个有限组
#     model_kmeans = KMeans(n_clusters=n_clusters)  # 建立聚类模型对象
#     labels_tmp = model_kmeans.fit_predict(X)  # 训练聚类模型
#     silhouette_tmp = silhouette_score(X, labels_tmp)  # 得到每个K下的平均轮廓系数
#     if silhouette_tmp > silhouette_int:  # 如果平均轮廓系数更高
#         best_k = n_clusters  # 保存K将最好的K存储下来
#         silhouette_int = silhouette_tmp  # 保存平均轮廓得分
#         best_kmeans = model_kmeans  # 保存模型实例对象
#         cluster_labels_k = labels_tmp  # 保存聚类标签
#     score_list.append([n_clusters, silhouette_tmp])  # 将每次K及其得分追加到列表
# # print('{:*^60}'.format('K值对应的轮廓系数:'))
# # print(np.array(score_list))  # 打印输出所有K下的详细得分
# # print('最优的K值是:{0} \n对应的轮廓系数是:{1}'.format(best_k, silhouette_int))
# # 将原始数据与聚类标签整合
# cluster_labels = pd.DataFrame(cluster_labels_k, columns=['clusters'])  # 获得训练集下的标签信息
# merge_data = pd.concat((raw_data2, cluster_labels), axis=1)  # 将原始处理过的数据跟聚类标签整合
# # print(merge_data.head())
# # 计算每个聚类类别下的样本量和样本占比
# clustering_count = pd.DataFrame(merge_data['渠道代号'].groupby(merge_data['clusters']).count()).T.rename({'渠道代号': 'counts'})  # 计算每个聚类类别的样本量
# clustering_ratio = (clustering_count / len(merge_data)).round(2).rename({'counts': 'percentage'})  # 计算每个聚类类别的样本量占比
# print(clustering_count)
# print("#"*30)
# print(clustering_ratio)
# # 计算每个聚类类别下的样本量和样本占比
# clustering_count = pd.DataFrame(merge_data['渠道代号'].groupby(merge_data['clusters']).count()).T.rename({'渠道代号': 'counts'})  # 计算每个聚类类别的样本量
# clustering_ratio = (clustering_count / len(merge_data)).round(2).rename({'counts': 'percentage'})  # 计算每个聚类类别的样本量占比
# # print(clustering_count)
# # print("#"*30)
# # print(clustering_ratio)
# # 计算各个聚类类别内部最显著特征值
# cluster_features = []  # 空列表，用于存储最终合并后的所有特征信息
# for line in range(best_k):  # 读取每个类索引
#     label_data = merge_data[merge_data['clusters'] == line]  # 获得特定类的数据
#
#     part1_data = label_data.iloc[:, 1:7]  # 获得数值型数据特征
#     part1_desc = part1_data.describe().round(3)  # 得到数值型特征的描述性统计信息
#     merge_data1 = part1_desc.iloc[2, :]  # 得到数值型特征的均值
#
#     part2_data = label_data.iloc[:, 7:-1]  # 获得字符串型数据特征
#     part2_desc = part2_data.describe(include='all')  # 获得字符串型数据特征的描述性统计信息
#     merge_data2 = part2_desc.iloc[2, :]  # 获得字符串型数据特征的最频繁值
#
#     merge_line = pd.concat((merge_data1, merge_data2), axis=0)  # 将数值型和字符串型典型特征沿行合并
#     cluster_features.append(merge_line)  # 将每个类别下的数据特征追加到列表
#
# #  输出完整的类别特征信息
# cluster_pd = pd.DataFrame(cluster_features).T  # 将列表转化为矩阵
# # print('{:*^60}'.format('每个类别主要的特征:'))
# all_cluster_set = pd.concat((clustering_count, clustering_ratio, cluster_pd),axis=0)  # 将每个聚类类别的所有信息合并
# # print(all_cluster_set)
# #各类别数据预处理
# num_sets = cluster_pd.iloc[:6, :].T.astype(np.float64)  # 获取要展示的数据
# num_sets_max_min = model_scaler.fit_transform(num_sets)  # 获得标准化后的数据
# print(num_sets)
# print('-'*20)
# print(num_sets_max_min)
# # 画图
# fig = plt.figure(figsize=(6,6))  # 建立画布
# ax = fig.add_subplot(111, polar=True)  # 增加子网格，注意polar参数
# labels = np.array(merge_data1.index)  # 设置要展示的数据标签
# cor_list = ['g', 'r', 'y', 'b']  # 定义不同类别的颜色
# angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)  # 计算各个区间的角度
# angles = np.concatenate((angles, [angles[0]]))  # 建立相同首尾字段以便于闭合
# # 画雷达图
# for i in range(len(num_sets)):  # 循环每个类别
#     data_tmp = num_sets_max_min[i, :]  # 获得对应类数据
#     data = np.concatenate((data_tmp, [data_tmp[0]]))  # 建立相同首尾字段以便于闭合
#     ax.plot(angles, data, 'o-', c=cor_list[i], label="第%d类渠道"%(i))  # 画线
#     ax.fill(angles, data,alpha=2.5)
# # 设置图像显示格式
# ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")  # 设置极坐标轴
# ax.set_title("各聚类类别显著特征对比", fontproperties="SimHei")  # 设置标题放置
# ax.set_rlim(-0.2, 1.2)  # 设置坐标轴尺度范围
# plt.legend(loc="upper right" ,bbox_to_anchor=(1.2,1.0))  # 设置图例位置
# plt.show()
from sklearn.datasets  import load_wine

load_wines=load_wine()
print(load_wines)
