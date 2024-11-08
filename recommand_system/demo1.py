# 导入所需的库
import json
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import seaborn as sns


# 生成随机用户数据的函数
def generate_user_data(num_users):
    user_data = []  # 初始化用户数据列表
    for _ in range(num_users):
        user = {
            "age": random.randint(18, 65),  # 随机年龄
            "gender": random.choice(["Male", "Female"]),  # 随机性别
            "is_married": random.choice([True, False]),  # 随机婚姻状态
            "phone_brand": random.choice(["Apple", "Samsung", "Huawei", "Xiaomi", "OnePlus"]),  # 随机手机品牌
            "salary": random.randint(30000, 120000),  # 随机薪水
            "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),  # 随机城市
            "education": random.choice(["High School", "Bachelor's", "Master's", "PhD"]),  # 随机教育程度
            "job_title": random.choice(["Engineer", "Doctor", "Teacher", "Artist", "Manager"]),  # 随机职业
            "daily_online_hours": random.uniform(0, 24),  # 随机每��在线小时数
            "username": f"user{_+1}"  # 生成用户名
        }
        user_data.append(user)  # 将用户添加到用户数据列表中
    return user_data  # 返回生成的用户数据

# 生成1000个用户           
users = generate_user_data(1000)

# 将用户数据保存到JSON文件
try:
    with open('./user_data.json', 'w') as json_file:
        json.dump(users, json_file, indent=4)  # 将用户数据写入JSON文件，缩进为4个空格
except Exception as e:
    print(f"发生错误: {e}")


def perform_kmeans_clustering(users, n_clusters=3):
    # 准备聚类数据
    X = [[user['age'], user['salary'], user['daily_online_hours']] for user in users]  # 提取年龄、薪水和每日在线小时数

    # 执行K均值聚类
    kmeans = KMeans(n_clusters=n_clusters)  # 创建K均值聚类对象，指定聚类数量
    kmeans.fit(X)  # 拟合数据
    labels = kmeans.labels_  # 获取每个用户的聚类标签

    # 绘制聚类图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.scatter([x[0] for x in X], [x[1] for x in X], c=labels, cmap='viridis', marker='o')  # 绘制散点图
    plt.title('User Clustering based on Age, Salary, and Daily Online Hours')  # 设置图表标题
    plt.xlabel('年龄')  # 设置X轴标签
    plt.ylabel('薪水')  # 设置Y轴标签
    plt.colorbar(label='聚类标签')  # 添加颜色条，标注聚类标签
    plt.show()  # 显示图形

# 调用函数进行聚类
# perform_kmeans_clustering(users, n_clusters=3)




from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

def find_similar_users(target_user, users):
    # 准备数据
    user_features = np.array([[user['age'], user['daily_online_hours']] for user in users])
    target_features = np.array([[target_user['age'],  target_user['daily_online_hours']]])

    # 计算余弦相似度
    similarities = cosine_similarity(target_features, user_features)[0]

    # 获取相似度最高的用户索引
    similar_user_indices = np.argsort(similarities)[::-1]

    return similar_user_indices, similarities

# 示例：查找与第一个用户相似的用户并绘制图表
target_user = users[0]
similar_user_indices, similarities = find_similar_users(target_user, users)

# 对相似度按从小到大排序
sorted_indices = np.argsort(similarities)
sorted_similarities = similarities[sorted_indices]


plt.figure(figsize=(10, 6))  # 设置图形大小
plt.bar(range(len(sorted_similarities)), sorted_similarities, color='skyblue')  # 绘制条形图
plt.title('Sorted Similarities of Users')  # 设置图表标题
plt.xlabel('用户索引')  # 设置X轴标签
plt.ylabel('相似度')  # 设置Y轴标签
plt.xticks(range(len(sorted_similarities)), sorted_indices)  # 设置X轴刻度为用户索引
plt.show()  # 显示图形

