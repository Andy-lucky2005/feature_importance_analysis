# main.py
# 执行完整分析流程

from analysis import run_analysis
if __name__ == "__main__":
    print("=" * 60)
    df_results = run_analysis()
    print("\n分析完成。结果保存在 'output' 文件夹中。")