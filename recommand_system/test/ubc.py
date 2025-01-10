
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# 加载数据
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('../ml-100k/u.data', sep='\t', names=column_names)
ratings.drop('timestamp', axis=1, inplace=True)

# 创建用户-电影评分矩阵
user_movie_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)


# 计算用户之间的余弦相似度
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def recommend_movies(user_id, user_movie_matrix, user_similarity_df, num_recommendations=10):
    # 获取用户的评分
    user_ratings = user_movie_matrix.loc[user_id]
    
    # 计算用户未评分的电影
    unrated_movies = user_ratings[user_ratings == 0].index
    
    # 计算推荐分数
    movie_scores = {}
    for movie in unrated_movies:
        sim_scores = user_similarity_df[user_id]
        movie_ratings = user_movie_matrix[movie]
        non_zero_indices = movie_ratings[movie_ratings > 0].index
        
        # 计算加权��均评分
        numerator = np.dot(sim_scores[non_zero_indices], movie_ratings[non_zero_indices])
        denominator = np.abs(sim_scores[non_zero_indices]).sum()
        
        if denominator > 0:
            movie_scores[movie] = numerator / denominator
    
    # 获取评分最高的电影
    recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    return [movie for movie, score in recommended_movies]

user_id = 1  # Replace with the desired user_id
recommended_movies = recommend_movies(user_id, user_movie_matrix, user_similarity_df)
print(f"Recommended movies for user {user_id}: {recommended_movies}")
