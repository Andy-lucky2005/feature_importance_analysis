# data_loader.py
# 读取 Excel，提取每个模型的排名矩阵

import pandas as pd
from config import DATA_FILE, MODEL_METHOD_KEYWORDS, FEATURE_NAMES

def load_rank_matrix():
    """加载 Excel，返回 DataFrame（行=特征，列=方法）"""
    df = pd.read_excel(DATA_FILE, index_col=0)
    # 确保索引与 FEATURE_NAMES 一致
    df.index = FEATURE_NAMES
    return df

def extract_model_rankings(df_rank):
    """
    根据 MODEL_METHOD_KEYWORDS 提取每个模型的排名矩阵
    返回字典: {model_name: numpy array (n_features, n_methods)}
    返回每一个模型的所有归因方法分析得出的特征重要性排序结果
    """
    model_matrices = {}
    for model, method_list in MODEL_METHOD_KEYWORDS.items():
        # 找出实际存在于 df_rank 中的列
        available = [col for col in method_list if col in df_rank.columns]
        # print(f"目前的方法为：{available}")

        if len(available) < 2:
            print(f"警告：模型 {model} 只有 {len(available)} 种方法，跳过")
            continue
        mat = df_rank[available].values
        model_matrices[model] = mat
        print(f"模型 {model}: 找到 {len(available)} 种方法，矩阵形状 {mat.shape}")
    return model_matrices