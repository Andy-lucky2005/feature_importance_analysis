import optuna
from sklearn.model_selection import KFold, cross_validate
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import time

# 设置固定的随机种子，确保每次结果一样
random_seed = 1412  # 固定随机种子

# 读取数据
file_path = "../Feature_Value/Science_feature_data.xlsx"
data = pd.read_excel(file_path, header=0)

# y 为第二列，目标变量
y = data.iloc[:, 1]  # 第二列作为目标变量 Eadh
# X 为从第三列到第16列的14个特征
X = data.iloc[:, 2:16]  # 从第三列到第16列（包含第16列）

'''贝叶斯随机森林寻优'''
# 定义随机森林回归模型的目标函数
def optuna_objective_rf(trial):
    # 选择超参数搜索空间
    n_estimators = trial.suggest_int('n_estimators', 10, 300, step= 1)  # 树的数量，取值范围从10到30
    max_depth = trial.suggest_int('max_depth', 3, 10, step= 1)  # 树的最大深度，取值范围从10到30
    max_features = trial.suggest_int('max_features', 1, 14, step= 1)  # 每棵树使用的最大特征数，取值范围从10到30
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    # 初始化随机森林回归模型
    reg = RandomForestRegressor(n_estimators=n_estimators,  # 树的数量
              max_depth=max_depth,  # 树的最大深度
              max_features=max_features,  # 每棵树使用的最大特征数
              min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf,
              random_state=random_seed,  # 固定随机种子，确保结果可复现
              verbose=False,  # 关闭详细输出
             )  # 使用所有可用的CPU核心进行并行计算

    # 使用10折交叉验证进行模型验证
    cv = KFold(n_splits=10, shuffle=True, random_state=random_seed)  # 10折交叉验证，数据会被随机打乱
    validation_loss = cross_validate(reg, X, y,
                                     scoring='neg_mean_squared_error',  # 使用负均方误差（越小越好）
                                     cv=cv,  # 交叉验证
     )  # 如果计算出现错误，则抛出异常
    score = np.mean(validation_loss['test_score'])
    print(f"Trial {trial.number}: score={score:.4f}, params={trial.params}")
    return np.mean(validation_loss['test_score'])  # 返回交叉验证的平均得分

# 定义优化过程，执行贝叶斯优化
def optimizer_optuna_rf(n_trials):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=30, n_ei_candidates=30, seed=random_seed),
                                direction='maximize')  # 创建Optuna优化研究对象，使用TPE采样器，固定优化种子
    study.optimize(optuna_objective_rf, n_trials=n_trials)  # 执行优化，优化目标是maximize（最大化）

    print('随机森林最优参数：\nbest_params:', study.best_trial.params,  # 输出最优参数
          '随机森林最优得分：\nbest_score:', study.best_trial.values,  # 输出最优得分
          '\n')

    return study.best_trial.params, study.best_trial.values  # 返回最优的超参数和得分

# 执行优化
optuna.logging.set_verbosity(optuna.logging.ERROR)  # 设置日志输出级别为ERROR，避免显示过多的信息
best_params_rf, best_score_rf = optimizer_optuna_rf(300)  # 执行100次优化试验

# 将最优参数和得分保存到文件
with open('RF_best_params.txt', 'w') as f:
    for key, value in best_params_rf.items():  # 保存最优超参数
        f.write(f'{key}: {value}\n')
    f.write(f'Best Score: {best_score_rf}\n')  # 保存最优得分

'''贝叶斯梯度提升树寻优'''
def optuna_objective_gbr(trial):
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
    n_estimators = trial.suggest_int('n_estimators', 10, 300)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    gbr = GradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,  # 设置固定的随机种子
    )

    cv = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    validation_loss = cross_validate(gbr, X, y,
                                     scoring='neg_mean_squared_error',
                                     cv=cv,
                                    )
    score = np.mean(validation_loss['test_score'])
    print(f"Trial {trial.number}: score={score:.4f}, params={trial.params}")
    return np.mean(validation_loss['test_score'])

# 定义优化目标函数
def optimizer_optuna_gbr(n_trials):
    # 通常建议 n_startup_trials ≈ 5~10 × 超参数个数，这样 TPE 可以建立比较可靠的先验。
    # 通常建议 n_ei_candidates ≈ 2~5 × 超参数个数
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=30, n_ei_candidates=30, seed=random_seed),
                                direction='maximize')
    study.optimize(optuna_objective_gbr, n_trials=n_trials)

    print('梯度提升树最优参数：\nbest_params:', study.best_trial.params,
          '梯度提升树最优得分：\nbest_score:', study.best_trial.value, '\n')

    return study.best_trial.params, study.best_trial.value

# 执行优化
best_params_gbr, best_score_gbr = optimizer_optuna_gbr(300)

# 保存最优参数和最优得分到文件
with open('GBRT_best_params.txt', 'w') as f:
    for key, value in best_params_gbr.items():
        f.write(f'{key}: {value}\n')
    f.write(f'Best Score: {best_score_gbr}\n')