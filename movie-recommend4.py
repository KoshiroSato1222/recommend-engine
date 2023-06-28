# ユーザーベースの協調フィルタリングで映画をリコメンドする
# pandasでデータを読み込む
# 最近傍探索で近い映画を探索する
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv("ml-latest-small/ratings.csv")

# 疎行列を作成する（ほとんどの要素が0の行列） アイテム/ユーザー行列
df_rating = df.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# コサイン類似度を求める
user_similarity = cosine_similarity(df_rating)

# DataFrameにする
user_sim_df = pd.DataFrame(user_similarity, index=df_rating.index, columns=df_rating.index)
print(user_sim_df.head())

print(user_sim_df[user_sim_df.index == 2].sort_values(by=2, axis=1, ascending=False))