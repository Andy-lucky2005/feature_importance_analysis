import pandas as pd
import numpy as np
import optuna
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate

# ---------------- 全局设置 ----------------
RANDOM_SEED = 1412
np.random.seed(RANDOM_SEED)

# ---------------- 读取数据 ----------------
file_path = "../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y：第2列（目标变量），X：第3~16列（14个特征）
y = data.iloc[:, 1]
X = data.iloc[:, 2:16]

# ---------------- KFold 交叉验证 ----------------
cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# ---------------- Optuna 目标函数（SVR） ----------------
def objective(trial):
    # 1. 超参数搜索空间（适配178×14的数据）
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
    C = trial.suggest_float('C', 1, 1000, log=True)  # 小样本回归，C一般不宜过大
    epsilon = trial.suggest_float('epsilon', 0.01, 0.2)  # 控制精度
    if kernel == 'rbf':
        gamma = trial.suggest_float('gamma', 1e-3, 1.0, log=True)
    else:
        gamma = 'scale'  # linear 核不需要 gamma

    # 2. 建立 Pipeline：标准化 + SVR
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma))
    ])

    # 3. 使用10折交叉验证
    scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring='neg_mean_squared_error'
    )
    mean_score = np.mean(scores['test_score'])

    print(f"Trial {trial.number}: score={mean_score:.4f}, params={trial.params}")
    return mean_score

# ---------------- 创建 Optuna Study（TPE 参数与XGBoost一致） ----------------
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=30,   # 与 XGBoost 一致
        n_ei_candidates=30,    # 与 XGBoost 一致
        seed=RANDOM_SEED
    ),
    direction='maximize'  # neg MSE 越大越好
)

# ---------------- 执行优化 ----------------
study.optimize(objective, n_trials=300)  # 200次足够探索这个规模的数据

# ---------------- 输出最优参数和分数 ----------------
print("\n===== SVR 最优结果 =====")
print("Best hyperparameters:", study.best_params)
print("Best negative MSE:", study.best_value)

with open('SVR_best_params.txt', 'w') as f:
    f.write("Best hyperparameters:\n")
    for key, value in study.best_params.items():
        f.write(f"{key}: {value}\n")
    f.write("\n")
    f.write(f"Best negative MSE: {study.best_value}\n")
