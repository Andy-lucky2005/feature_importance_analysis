import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import optuna

# ---------------- 全局设置 ----------------
random_seed = 1412
np.random.seed(random_seed)

# ---------------- 读取数据 ----------------
file_path = "../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)
y = data.iloc[:, 1]
X = data.iloc[:, 2:16]

# ---------------- 定义优化目标函数 ----------------
def objective(trial):
    # 隐藏层层数 1~2 层
    n_layers = trial.suggest_int('n_layers', 1, 2)
    hidden_layer_sizes = []
    for i in range(n_layers):
        if i == 0:
            n_units = trial.suggest_int(f'n_units_l{i}', 10, 50)  # 第一层神经元
        else:
            n_units = trial.suggest_int(f'n_units_l{i}', 5, 30)   # 第二层神经元
        hidden_layer_sizes.append(n_units)
    hidden_layer_sizes = tuple(hidden_layer_sizes)

    # 激活函数、优化器、正则化
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    solver = trial.suggest_categorical('solver', ['adam', 'lbfgs'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)

    # 学习率与早停
    if solver == 'adam':
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        early_stopping = True
        max_iter = 5000  # adam 配合早停，迭代次数可适中
    else:
        learning_rate_init = 0.001
        early_stopping = False
        max_iter = 2000  # 限制 lbfgs 最大迭代次数，防止过拟合

    # ---------------- 构建 Pipeline ----------------
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            random_state=random_seed,
            early_stopping=early_stopping,
            n_iter_no_change=20
        ))
    ])

    # ---------------- 交叉验证 ----------------
    cv = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    scores = cross_validate(model, X, y,
                             cv=cv,
                             scoring='neg_mean_squared_error'
                            )

    return np.mean(scores['test_score'])

# ---------------- 创建 Optuna study ----------------
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=30,   # 前25次随机探索
        n_ei_candidates=30,    # 每次选择候选数
        seed=random_seed
    ),
    direction='maximize'
)

# ---------------- 执行超参数优化 ----------------
study.optimize(objective, n_trials=300)

# ---------------- 输出与保存结果 ----------------
print("Best hyperparameters:", study.best_params)
print("Best negative MSE: {:.6f}".format(study.best_value))

with open('MLP_best_params.txt', 'w') as f:
    f.write("Best hyperparameters:\n")
    for key, value in study.best_params.items():
        f.write(f"{key}: {value}\n")
    f.write("\n")
    f.write(f"Best negative MSE: {study.best_value}\n")
