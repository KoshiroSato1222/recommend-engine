from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

A = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]])
print(cosine_similarity(A))