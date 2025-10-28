import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------- 数据读取 --------------------
file_path = "../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y 为第二列
y = data.iloc[:, 1]
# X 为从第三列到第16列的14个特征
X = data.iloc[:, 2:16]

# 固定随机种子
random_seed = 1412

# -------------------- Optuna 目标函数 --------------------
def objective(trial):
    # 定义要优化的超参数
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 10, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'random_state': random_seed
    }

    # 使用 Pipeline 包含标准化
    model = Pipeline([
        ('scaler', StandardScaler()),       # 标准化处理
        ('xgb', xgb.XGBRegressor(**param))  # XGBoost 回归器
    ])

    # 10折交叉验证
    cv = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring='neg_mean_squared_error'
    )
    mean_score = np.mean(scores['test_score'])
    print(f"Trial {trial.number}: score={mean_score:.4f}, params={trial.params}")
    return mean_score

# -------------------- 创建 Optuna 学习任务 --------------------
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=30,
        n_ei_candidates=30,
        seed=random_seed
    ),
    direction='maximize'
)

# 执行优化
study.optimize(objective, n_trials=300)

# -------------------- 输出最优参数和得分 --------------------
print("Best hyperparameters: ", study.best_params)
print("Best score: {:.4f}".format(study.best_value))

# -------------------- 保存结果 --------------------
with open('XGBoost_best_params.txt', 'w') as f:
    for key, value in study.best_params.items():
        f.write(f'{key}: {value}\n')
    f.write(f'Best Score: {study.best_value}\n')
