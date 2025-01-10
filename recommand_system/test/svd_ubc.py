from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def svd_recommend_movies(user_id, user_movie_matrix, num_recommendations=10):
    # 使用SVD进行降维
    svd = TruncatedSVD(n_components=50)  # 选择50个潜在特征
    user_movie_matrix_svd = svd.fit_transform(user_movie_matrix)

    # 计算用户之间的相似度
    user_similarity = cosine_similarity(user_movie_matrix_svd)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

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
        
        # 计算加权平均评分
        numerator = np.dot(sim_scores[non_zero_indices], movie_ratings[non_zero_indices])
        denominator = np.abs(sim_scores[non_zero_indices]).sum()
        
        if denominator > 0:
            movie_scores[movie] = numerator / denominator
    
    # 获取评分最高的电影
    recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    return [movie for movie, score in recommended_movies]



# 加载数据
ratings_df = pd.read_csv('../ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
ratings_df.drop('timestamp', axis=1, inplace=True)

# 创建用户-电影评分矩阵
user_movie_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)


# 示例用户推荐
user_id = 1  # 替换为所需的用户ID
recommended_movies = svd_recommend_movies(user_id, user_movie_matrix)
print(f"Recommended movies for user {user_id} using SVD: {recommended_movies}")

