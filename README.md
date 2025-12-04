# feature\_importance\_analysis



This repository contains all code and data structures related to feature importance evaluation, model performance analysis, and result visualization for the paper “Rethinking Feature Importance for Interpretable Machine Learning in Catalyst Design.”



The original dataset used in the study is provided in Feature\_Value/Science\_feature\_data.xlsx.



📂 Project Structure

feature\_importance\_analysis/

│

├── Feature\_Value/  

│   └── Analyze the value range of each feature and perform Z-score standardization

│

├── GA\_algorithm/

│   └── Feature ranking using a genetic algorithm optimization approach

│

├── Hyperparameter\_optimization/

│   └── Scripts and results for machine learning model hyperparameter optimization

│

├── Linear\_regression\_coefficient/

│   └── Feature importance analysis based on linear model coefficients

│

├── MDI/

│   └── Mean Decrease in Impurity (tree-based feature importance)

│

├── Model\_free\_analysis/

│   └── Model-independent feature importance methods (Pearson, Spearman, Mutual Information)

│

├── SHAP/

│   └── SHAP (SHapley Additive exPlanations) analysis and visualization

│

├── SISSO\_Formual\_analysis/

│   └── SISSO (Sure Independence Screening and Sparsifying Operator) formula extraction and interpretation

│

├── Test\_statistic/

│   └── Statistical tests:

│        - H1 hypothesis testing for cross-method consistency  

│        - Feature importance significance test for leading-feature dominance

│

├── feature\_importance\_summary/

│   └── Comprehensive summary and comparison of feature importance across all methods

│

├── model\_performance\_analysis/

│   └── Evaluation of model performance (MAE, R²)

│

└── permutation\_importance/

&nbsp;   └── Permutation Importance analysis and results



