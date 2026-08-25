# -*- coding: utf-8 -*-
"""
银行贷款审批预测：Baseline 建模 V1.0
==================================

用途：
- 基于 V1.4 已冻结的 train_processed_v14.csv / valid_processed_v14.csv
  进行第一轮 Baseline 模型横向比较。
- 本阶段不做 SMOTE、不做类别权重、不做系统调参。
- 统一以 ROC-AUC 为主，PR-AUC 为重要参考。
- 同时记录默认 0.5 阈值指标与“验证集最佳 F1 阈值”，
  但最佳 F1 阈值只用于探索，不作为最终业务阈值。

模型：
1. Logistic Regression
   - 数值特征 StandardScaler
   - 类别特征 One-Hot Encoding
2. Random Forest
3. LightGBM
   - 显式标记 categorical features
4. XGBoost
   - 第一轮基线把 V1.4 数值编码后的类别作为数值输入
5. CatBoost
   - 显式标记 categorical features

重要：
- 请把 RANDOM_STATE 改成你项目要求的“学号随机种子”。
- Test 数据只用于列结构检查，不参与模型选择、阈值选择或调参。
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# 第三方模型
try:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError(
        "缺少建模依赖。请先执行：\n"
        "pip install lightgbm xgboost catboost scikit-learn pandas numpy matplotlib\n"
        f"\n原始错误：{e}"
    )

import matplotlib.pyplot as plt


# ============================================================
# 1. 配置区
# ============================================================

DATA_DIR = r"D:\深圳点宽\银行贷款审批\output_v14"

TRAIN_PATH = os.path.join(DATA_DIR, "train_processed_v14.csv")
VALID_PATH = os.path.join(DATA_DIR, "valid_processed_v14.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_processed_v14.csv")

OUTPUT_DIR = os.path.join(DATA_DIR, "baseline_v1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# !!! 请改成你项目要求的“学号随机种子” !!!
RANDOM_STATE = 42

TARGET = "Approved"
ID_COL = "ID"

CATEGORICAL_FEATURES = [
    "Gender",
    "City_Category",
    "Employer_Category1",
    "Employer_Category2",
    "Customer_Existing_Primary_Bank_Code",
    "Primary_Bank_Type",
    "Source",
    "Source_Category",
    "City_Code",
    "Age_Bin",
]


# ============================================================
# 2. 工具函数
# ============================================================

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    return pd.read_csv(path)


def calc_best_f1_threshold(y_true, prob):
    """
    在验证集上寻找 F1 最大的概率阈值。
    仅用于第一轮探索，不视为最终业务阈值。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)

    if len(thresholds) == 0:
        return {
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # precision / recall 比 thresholds 多一个点，因此排除最后一个点
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    f1_values = (
        2 * precision_t * recall_t
        / (precision_t + recall_t + 1e-12)
    )

    best_idx = int(np.nanargmax(f1_values))

    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision_t[best_idx]),
        "recall": float(recall_t[best_idx]),
        "f1": float(f1_values[best_idx]),
    }


def evaluate_model(model_name, y_true, prob, train_seconds):
    """
    统一评估口径。
    """
    roc_auc = roc_auc_score(y_true, prob)
    pr_auc = average_precision_score(y_true, prob)

    # 默认阈值 0.5
    pred_05 = (prob >= 0.5).astype(int)

    p05 = precision_score(
        y_true, pred_05, zero_division=0
    )
    r05 = recall_score(
        y_true, pred_05, zero_division=0
    )
    f105 = f1_score(
        y_true, pred_05, zero_division=0
    )

    cm05 = confusion_matrix(
        y_true, pred_05
    )
    tn05, fp05, fn05, tp05 = cm05.ravel()

    # 探索性最佳 F1 阈值
    best = calc_best_f1_threshold(
        y_true, prob
    )

    pred_best = (
        prob >= best["threshold"]
    ).astype(int)

    cm_best = confusion_matrix(
        y_true, pred_best
    )
    tnb, fpb, fnb, tpb = cm_best.ravel()

    row = {
        "Model": model_name,

        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,

        "Precision@0.5": p05,
        "Recall@0.5": r05,
        "F1@0.5": f105,

        "TN@0.5": int(tn05),
        "FP@0.5": int(fp05),
        "FN@0.5": int(fn05),
        "TP@0.5": int(tp05),

        "Best_F1_Threshold": best["threshold"],
        "Precision@BestF1": best["precision"],
        "Recall@BestF1": best["recall"],
        "Best_F1": best["f1"],

        "TN@BestF1": int(tnb),
        "FP@BestF1": int(fpb),
        "FN@BestF1": int(fnb),
        "TP@BestF1": int(tpb),

        "Train_Seconds": train_seconds,
    }

    return row


def print_model_result(row):
    print(
        f"\n[{row['Model']}]"
        f"\n  ROC-AUC : {row['ROC_AUC']:.6f}"
        f"\n  PR-AUC  : {row['PR_AUC']:.6f}"
        f"\n  @0.5    : "
        f"Precision={row['Precision@0.5']:.6f}, "
        f"Recall={row['Recall@0.5']:.6f}, "
        f"F1={row['F1@0.5']:.6f}"
        f"\n  Best F1 : "
        f"threshold={row['Best_F1_Threshold']:.6f}, "
        f"Precision={row['Precision@BestF1']:.6f}, "
        f"Recall={row['Recall@BestF1']:.6f}, "
        f"F1={row['Best_F1']:.6f}"
        f"\n  训练耗时 : {row['Train_Seconds']:.3f} 秒"
    )


# ============================================================
# 3. 加载数据
# ============================================================

print("=" * 78)
print("银行贷款审批预测：Baseline 建模 V1.0")
print("=" * 78)

train = load_csv(TRAIN_PATH)
valid = load_csv(VALID_PATH)
test = load_csv(TEST_PATH)

print(f"Train: {train.shape}")
print(f"Valid: {valid.shape}")
print(f"Test : {test.shape}")

if TARGET not in train.columns:
    raise ValueError("训练集缺少 Approved")

if TARGET not in valid.columns:
    raise ValueError("验证集缺少 Approved")

if ID_COL not in test.columns:
    raise ValueError("测试集缺少 ID")

feature_cols = [
    c for c in train.columns
    if c != TARGET
]

valid_feature_cols = [
    c for c in valid.columns
    if c != TARGET
]

test_feature_cols = [
    c for c in test.columns
    if c != ID_COL
]

if feature_cols != valid_feature_cols:
    raise ValueError(
        "Train / Valid 特征列不一致"
    )

if feature_cols != test_feature_cols:
    raise ValueError(
        "Train / Test 特征列不一致"
    )

missing_cat_cols = [
    c for c in CATEGORICAL_FEATURES
    if c not in feature_cols
]

if missing_cat_cols:
    raise ValueError(
        f"缺少类别特征：{missing_cat_cols}"
    )

numeric_features = [
    c for c in feature_cols
    if c not in CATEGORICAL_FEATURES
]

X_train = train[feature_cols].copy()
y_train = train[TARGET].astype(int).copy()

X_valid = valid[feature_cols].copy()
y_valid = valid[TARGET].astype(int).copy()

X_test = test[feature_cols].copy()

print(
    f"\n训练集正样本：{int(y_train.sum())} / {len(y_train)} "
    f"({y_train.mean() * 100:.6f}%)"
)

print(
    f"验证集正样本：{int(y_valid.sum())} / {len(y_valid)} "
    f"({y_valid.mean() * 100:.6f}%)"
)

print(f"\n数值特征数：{len(numeric_features)}")
print(f"类别特征数：{len(CATEGORICAL_FEATURES)}")
print(f"总特征数：{len(feature_cols)}")

if (
    X_train.isna().sum().sum() != 0
    or X_valid.isna().sum().sum() != 0
    or X_test.isna().sum().sum() != 0
):
    raise ValueError(
        "发现 NaN，请先检查 V1.4 预处理"
    )


# ============================================================
# 4. Baseline 训练
# ============================================================

results = []
valid_predictions = pd.DataFrame({
    "y_true": y_valid.values
})


# ------------------------------------------------------------
# 4.1 Logistic Regression
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("1/5 Logistic Regression")

# 数值 -> 标准化
# 类别 -> One-Hot
logistic_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES
        ),
    ]
)

logistic_model = Pipeline(
    steps=[
        (
            "preprocess",
            logistic_preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE
            )
        ),
    ]
)

start = time.time()
logistic_model.fit(
    X_train,
    y_train
)
elapsed = time.time() - start

prob = logistic_model.predict_proba(
    X_valid
)[:, 1]

row = evaluate_model(
    "Logistic Regression",
    y_valid,
    prob,
    elapsed
)

results.append(row)
valid_predictions[
    "Logistic_Regression"
] = prob

print_model_result(row)


# ------------------------------------------------------------
# 4.2 Random Forest
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("2/5 Random Forest")

rf_model = RandomForestClassifier(
    n_estimators=400,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

start = time.time()
rf_model.fit(
    X_train,
    y_train
)
elapsed = time.time() - start

prob = rf_model.predict_proba(
    X_valid
)[:, 1]

row = evaluate_model(
    "Random Forest",
    y_valid,
    prob,
    elapsed
)

results.append(row)
valid_predictions[
    "Random_Forest"
] = prob

print_model_result(row)


# ------------------------------------------------------------
# 4.3 LightGBM
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("3/5 LightGBM")

# 显式转为 pandas category。
# V1.4 的未知类别已经编码为 -1。
X_train_lgb = X_train.copy()
X_valid_lgb = X_valid.copy()

for col in CATEGORICAL_FEATURES:

    # 加上 -1，保证未来未知类别可被识别
    categories = sorted(
        set(
            X_train_lgb[col]
            .astype(int)
            .tolist()
        )
        | {-1}
    )

    X_train_lgb[col] = pd.Categorical(
        X_train_lgb[col].astype(int),
        categories=categories
    )

    X_valid_lgb[col] = pd.Categorical(
        X_valid_lgb[col].astype(int),
        categories=categories
    )

lgb_model = LGBMClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

start = time.time()
lgb_model.fit(
    X_train_lgb,
    y_train,
    categorical_feature=CATEGORICAL_FEATURES
)
elapsed = time.time() - start

prob = lgb_model.predict_proba(
    X_valid_lgb
)[:, 1]

row = evaluate_model(
    "LightGBM",
    y_valid,
    prob,
    elapsed
)

results.append(row)
valid_predictions[
    "LightGBM"
] = prob

print_model_result(row)


# ------------------------------------------------------------
# 4.4 XGBoost
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("4/5 XGBoost")

# Baseline 第一轮：
# 暂时将 V1.4 已数值编码的类别字段作为数值输入。
# 后续若 XGBoost 值得继续优化，再专门测试原生类别/One-Hot 等方案。
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

start = time.time()
xgb_model.fit(
    X_train,
    y_train
)
elapsed = time.time() - start

prob = xgb_model.predict_proba(
    X_valid
)[:, 1]

row = evaluate_model(
    "XGBoost",
    y_valid,
    prob,
    elapsed
)

results.append(row)
valid_predictions[
    "XGBoost"
] = prob

print_model_result(row)


# ------------------------------------------------------------
# 4.5 CatBoost
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("5/5 CatBoost")

# CatBoost 将类别列转为字符串，
# 防止模型把类别编码误解为连续数值大小。
X_train_cat = X_train.copy()
X_valid_cat = X_valid.copy()

for col in CATEGORICAL_FEATURES:
    X_train_cat[col] = (
        X_train_cat[col]
        .astype(str)
    )
    X_valid_cat[col] = (
        X_valid_cat[col]
        .astype(str)
    )

cat_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=False
)

start = time.time()
cat_model.fit(
    X_train_cat,
    y_train,
    cat_features=CATEGORICAL_FEATURES
)
elapsed = time.time() - start

prob = cat_model.predict_proba(
    X_valid_cat
)[:, 1]

row = evaluate_model(
    "CatBoost",
    y_valid,
    prob,
    elapsed
)

results.append(row)
valid_predictions[
    "CatBoost"
] = prob

print_model_result(row)


# ============================================================
# 5. 汇总结果
# ============================================================

results_df = pd.DataFrame(
    results
)

# 主指标 ROC-AUC 排序
results_df = (
    results_df
    .sort_values(
        ["ROC_AUC", "PR_AUC"],
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 78)
print("Baseline 模型总表（按 ROC-AUC 排序）")
print("=" * 78)

display_cols = [
    "Model",
    "ROC_AUC",
    "PR_AUC",
    "Precision@0.5",
    "Recall@0.5",
    "F1@0.5",
    "Best_F1_Threshold",
    "Precision@BestF1",
    "Recall@BestF1",
    "Best_F1",
    "Train_Seconds",
]

print(
    results_df[
        display_cols
    ].to_string(index=False)
)


# ============================================================
# 6. 保存结果
# ============================================================

result_path = os.path.join(
    OUTPUT_DIR,
    "baseline_results.csv"
)

pred_path = os.path.join(
    OUTPUT_DIR,
    "baseline_valid_predictions.csv"
)

results_df.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig"
)

valid_predictions.to_csv(
    pred_path,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\n✅ 已保存：{result_path}"
)
print(
    f"✅ 已保存：{pred_path}"
)


# ============================================================
# 7. ROC 曲线
# ============================================================

plt.figure(figsize=(8, 6))

for col in valid_predictions.columns:
    if col == "y_true":
        continue

    fpr, tpr, _ = roc_curve(
        y_valid,
        valid_predictions[col]
    )

    auc = roc_auc_score(
        y_valid,
        valid_predictions[col]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{col} (AUC={auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Baseline ROC Curves")
plt.legend()
plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR,
    "baseline_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=160
)
plt.close()

print(
    f"✅ 已保存：{roc_path}"
)


# ============================================================
# 8. PR 曲线
# ============================================================

plt.figure(figsize=(8, 6))

for col in valid_predictions.columns:
    if col == "y_true":
        continue

    precision, recall, _ = precision_recall_curve(
        y_valid,
        valid_predictions[col]
    )

    ap = average_precision_score(
        y_valid,
        valid_predictions[col]
    )

    plt.plot(
        recall,
        precision,
        label=f"{col} (AP={ap:.4f})"
    )

baseline_rate = y_valid.mean()

plt.axhline(
    baseline_rate,
    linestyle="--",
    label=f"Positive rate={baseline_rate:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Baseline Precision-Recall Curves")
plt.legend()
plt.tight_layout()

pr_path = os.path.join(
    OUTPUT_DIR,
    "baseline_pr_curve.png"
)

plt.savefig(
    pr_path,
    dpi=160
)
plt.close()

print(
    f"✅ 已保存：{pr_path}"
)


# ============================================================
# 9. 运行摘要
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "baseline_summary.txt"
)

best_roc = results_df.iloc[0]

best_pr = (
    results_df
    .sort_values(
        "PR_AUC",
        ascending=False
    )
    .iloc[0]
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "银行贷款审批预测 - Baseline V1.0\n"
    )
    f.write("=" * 70 + "\n")

    f.write(
        f"RANDOM_STATE = {RANDOM_STATE}\n"
    )

    f.write(
        f"Train = {X_train.shape}, "
        f"正样本率={y_train.mean():.8f}\n"
    )

    f.write(
        f"Valid = {X_valid.shape}, "
        f"正样本率={y_valid.mean():.8f}\n"
    )

    f.write(
        "\n按 ROC-AUC 排序：\n"
    )

    f.write(
        results_df[
            [
                "Model",
                "ROC_AUC",
                "PR_AUC",
                "Best_F1_Threshold",
                "Best_F1"
            ]
        ].to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "ROC-AUC 最佳模型："
        f"{best_roc['Model']} "
        f"({best_roc['ROC_AUC']:.6f})\n"
    )

    f.write(
        "PR-AUC 最佳模型："
        f"{best_pr['Model']} "
        f"({best_pr['PR_AUC']:.6f})\n"
    )

    f.write(
        "\n说明：\n"
        "1. 本轮未做 SMOTE / class_weight / scale_pos_weight 等不平衡处理。\n"
        "2. Best F1 threshold 仅为验证集探索结果，不是最终业务阈值。\n"
        "3. Test 未参与模型选择、调参或阈值确定。\n"
    )

print(
    f"✅ 已保存：{summary_path}"
)

print("\n" + "=" * 78)
print("Baseline V1.0 完成")
print("=" * 78)

print(
    "\n请把 baseline_v1 文件夹中的以下文件发给我："
)
print(
    "1. baseline_results.csv"
)
print(
    "2. baseline_valid_predictions.csv"
)
print(
    "3. baseline_summary.txt"
)
print(
    "4. baseline_roc_curve.png"
)
print(
    "5. baseline_pr_curve.png"
)
