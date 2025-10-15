import re
import math
import numpy as np
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Stemmer and Stop Words
STEMMER = PorterStemmer()
STOP_WORDS = stopwords.words("english")

# Constants
DOC_SIZE = 1095

# 文檔前處理函數
def doc_preprocessing(doc: str) -> list:
    doc = re.sub(r"\s+", " ", doc.replace("\n", "").replace("\r", ""))
    doc = re.sub(r"[^\w\s]", "", doc.lower())
    words = doc.split()
    words = [STEMMER.stem(word) for word in words if word not in STOP_WORDS]
    return words

# 計算 TF 和 DF
def get_tf_and_df(corpus: list):
    tf_list = []
    df_dict = {}
    for document_id, document in corpus:
        words = doc_preprocessing(document)
        tf = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1
        tf_list.append((document_id, tf))
        for term in tf:
            df_dict[term] = df_dict.get(term, 0) + 1
    return tf_list, dict(sorted(df_dict.items()))

# 建立 Index Dictionary
def get_index_dict(df_dict: dict) -> dict:
    return {term: idx for idx, term in enumerate(df_dict)}

# 計算 TF 和 TF-IDF 向量
def get_tf_vector(tf_list, index_dict):
    tf_vectors = []
    for document_id, tf_dict in tf_list:
        tf_vector = np.zeros(len(index_dict))
        for term, count in tf_dict.items():
            tf_vector[index_dict[term]] = count
        tf_vectors.append((document_id, tf_vector))
    return tf_vectors

def get_tf_idf_vector(tf_vectors, df_dict, index_dict):
    idf_vector = np.array([math.log(DOC_SIZE / df, 10) for df in df_dict.values()])
    tf_idf_vectors = []
    for document_id, tf_vector in tf_vectors:
        tf_idf = tf_vector * idf_vector
        tf_idf_vectors.append((document_id, tf_idf / np.linalg.norm(tf_idf)))
    return tf_idf_vectors

# 計算 Cosine Similarity 矩陣
def cosine_matrix(doc_vectors):
    norm = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    normalized_vectors = doc_vectors / norm
    return np.dot(normalized_vectors, normalized_vectors.T)

# HAC 聚類
def hac_clustering(C, k_values):
    I = np.ones(DOC_SIZE, dtype=bool)
    hac_dict = {i: [i] for i in range(DOC_SIZE)}
    merge_list = []
    
    while len(hac_dict) > min(k_values):
        # 找到相似度最高的兩個聚類
        max_sim = -1
        i, j = -1, -1
        for x in hac_dict:
            for y in hac_dict:
                if x != y and C[x][y] > max_sim:
                    max_sim, i, j = C[x][y], x, y

        # 合併兩個聚類
        hac_dict[i].extend(hac_dict.pop(j))
        I[j] = False
        
        # 更新相似度矩陣
        for z in hac_dict:
            if z != i:
                C[i][z] = C[z][i] = min(C[i][z], C[j][z])

        merge_list.append((i, j))
        if len(hac_dict) in k_values:
            write_result(hac_dict, len(hac_dict))

# 保存結果
def write_result(hac_dict, cluster_num):
    with open(f"{cluster_num}.txt", "w") as f:
        for cluster in sorted(hac_dict.values()):
            for doc_id in sorted(cluster):
                f.write(f"{doc_id + 1}\n")
            f.write("\n")

# 主程序
if __name__ == "__main__":
    # 讀取文檔
    corpus_path = "./data/IRTM/"
    files = sorted([f for f in os.listdir(corpus_path) if f.endswith(".txt")], key=lambda x: int(x[:-4]))
    corpus = [(int(f[:-4]), open(os.path.join(corpus_path, f)).read()) for f in files]

    # 計算 TF 和 DF
    tf_list, df_dict = get_tf_and_df(corpus)
    index_dict = get_index_dict(df_dict)
    tf_vectors = get_tf_vector(tf_list, index_dict)
    tf_idf_vectors = np.array([vec for _, vec in get_tf_idf_vector(tf_vectors, df_dict, index_dict)])
    
    # 計算 Cosine 相似度
    similarity_matrix = cosine_matrix(tf_idf_vectors)

    # HAC 聚類
    hac_clustering(similarity_matrix, [8, 13, 20])
