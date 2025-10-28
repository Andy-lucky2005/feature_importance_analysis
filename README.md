# Explainable analysis
For paper 'Rethinking feature importance for interpretable machine learning in catalyst design'

## Repository Structure

- **Feature_Value**
  Contains raw and preprocessed feature values from the dataset 
  (Science_feature_data.xlsx), used for all subsequent analyses.

- **Hyperparameter_optimization**  
  Scripts for tuning model hyperparameters.

- **Linear_regression_coefficient**  
  Computes linear regression coefficients to assess the linear contribution of features.

- **MDI**  
  Implements Mean Decrease Impurity (MDI) feature importance using tree-based models 
  (Random Forest, GBRT, XGBoost).

- **Model_free_analysis**  
  Model-agnostic feature evaluation using Pearson correlation, Spearman correlation, 
  and Mutual Information (MI).

- **SHAP**  
  Explains model predictions with SHAP (SHapley Additive exPlanations), supporting both 
  KernelSHAP and TreeSHAP.
  
- **SISSO_Formula_analysis**
  Feature analysis based on SISSO-generated formulas:
﻿
  performance – SISSO model performance evaluation
﻿
  AGM – Average Gradient Magnitude analysis
﻿
  SGR – Sample Gradient Ranking analysis
﻿
  SHAP – SHAP-based feature contribution analysis

- **Spearman_all_methods**  
  Spearman correlation analysis across all feature importance methods to assess 
  consistency.

- **feature_importance_summary**  
  Aggregates feature importance results from all methods and generates visualizations.

- **model_performance_analysis**  
  Analyzes model performance metrics such as MAE and R².

- **permutation_importance**  
  Computes permutation feature importance (PFI) for evaluating feature contributions.

---

## 使用说明

1. Place the original dataset Science_feature_data.xlsx in the Feature_Value folder.
2. Run the scripts corresponding to the analysis method or model you want to use.  

---
