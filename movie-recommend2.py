# ユーザーベースの協調フィルタリングで映画をリコメンドする
# pandasでデータを読み込む
# 最近傍探索で近い映画を探索する
import pandas as pd
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("ml-latest-small/ratings.csv")

# 疎行列を作成する（ほとんどの要素が0の行列） アイテム/ユーザー行列
df_rating = df.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# 最近傍探索の実行
neigh = NearestNeighbors(metric="cosine")
# 学習する
neigh.fit(df_rating)

# userId=２に似ている映画に近いものを探索する
distance, indices = neigh.kneighbors(df_rating[df_rating.index == 2])

# 二次元リストの（インデックス）が得られた
print(indices)

# 一次元リストになる
print(indices.flatten())

# movieIdを表示する
for i in indices.flatten():
    print(df_rating.index[i])


# 近い人の評価点が高いものを表示する
movies_list = df[df["userId"] == 366]
# 点数の高い順に表示する
print(movies_list.sort_values(by="rating", ascending=False))