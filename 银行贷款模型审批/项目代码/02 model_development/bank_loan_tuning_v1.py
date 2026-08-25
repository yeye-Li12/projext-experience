# -*- coding: utf-8 -*-
"""
银行贷款审批预测：模型参数调优 V1.0
================================

固定数据版本：
- train_processed_v14.csv
- valid_processed_v14.csv
- test_processed_v14.csv

固定随机种子：
RANDOM_STATE = 42

本阶段承接类别不平衡实验结论：
1. CatBoost：
   - 主排名 ROC-AUC 最佳策略仍为 Baseline（无类别权重）
   - 因此本轮主要调模型容量、学习率和正则，不继续加大类别权重

2. XGBoost：
   - scale_pos_weight=20 明显优于原始 Baseline
   - 因此本轮固定 scale_pos_weight=20，再调树结构和正则

3. Logistic Regression：
   - class_weight='balanced' 明显损害 ROC-AUC / PR-AUC
   - 因此只保留无权重版本，调 C 作为解释型基准

原则：
- 不使用 Test 进行参数选择
- 不做大规模暴力网格搜索
- 使用一组事先定义的小规模候选参数，降低对单一 Validation 的过拟合风险
- ROC-AUC 为主排名指标，PR-AUC 为重要参考
- Best-F1 阈值只用于探索，不是最终业务阈值
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
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

try:
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError(
        "缺少依赖，请先执行：\n"
        "pip install xgboost catboost scikit-learn pandas numpy\n"
        f"\n原始错误：{e}"
    )


# ============================================================
# 1. 配置
# ============================================================

DATA_DIR = r"D:\深圳点宽\银行贷款审批\output_v14"

TRAIN_PATH = os.path.join(DATA_DIR, "train_processed_v14.csv")
VALID_PATH = os.path.join(DATA_DIR, "valid_processed_v14.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_processed_v14.csv")

OUTPUT_DIR = os.path.join(DATA_DIR, "tuning_v1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# 2. 候选参数
# ============================================================

# Logistic：仅调正则强度，不使用 class_weight
LR_C_VALUES = [
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]

# CatBoost：
# C00 为当前正式 Baseline 的参数，用来核对结果；
# 后续配置主要围绕更低学习率、不同深度、L2 正则进行小范围搜索。
CAT_CONFIGS = [
    {
        "name": "CAT_C00_Baseline",
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.10,
        "l2_leaf_reg": 3,
    },
    {
        "name": "CAT_C01",
        "iterations": 500,
        "depth": 5,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
    },
    {
        "name": "CAT_C02",
        "iterations": 600,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
    },
    {
        "name": "CAT_C03",
        "iterations": 800,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5,
    },
    {
        "name": "CAT_C04",
        "iterations": 800,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5,
    },
    {
        "name": "CAT_C05",
        "iterations": 500,
        "depth": 7,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
    },
    {
        "name": "CAT_C06",
        "iterations": 700,
        "depth": 4,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5,
    },
    {
        "name": "CAT_C07",
        "iterations": 700,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 10,
    },
    {
        "name": "CAT_C08",
        "iterations": 600,
        "depth": 5,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10,
    },
    {
        "name": "CAT_C09",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10,
    },
]

# XGBoost：
# 固定 scale_pos_weight=20（上一阶段当前最佳）
# X00 为上一阶段正式参数，用于核对。
XGB_CONFIGS = [
    {
        "name": "XGB_X00_SPW20_Baseline",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.10,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "XGB_X01",
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "XGB_X02",
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "XGB_X03",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_lambda": 3.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "XGB_X04",
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "XGB_X05",
        "n_estimators": 700,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "gamma": 0.1,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "XGB_X06",
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.1,
        "reg_lambda": 5.0,
        "reg_alpha": 0.1,
    },
    {
        "name": "XGB_X07",
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.04,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "gamma": 0.2,
        "reg_lambda": 10.0,
        "reg_alpha": 0.1,
    },
    {
        "name": "XGB_X08",
        "n_estimators": 800,
        "max_depth": 3,
        "learning_rate": 0.025,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.1,
        "reg_lambda": 10.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "XGB_X09",
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "gamma": 0.2,
        "reg_lambda": 10.0,
        "reg_alpha": 0.5,
    },
]


# ============================================================
# 3. 工具函数
# ============================================================

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    return pd.read_csv(path)


def best_f1_info(y_true, prob):
    precision, recall, thresholds = precision_recall_curve(y_true, prob)

    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0

    p = precision[:-1]
    r = recall[:-1]

    f1 = (
        2 * p * r
        / (p + r + 1e-12)
    )

    idx = int(np.nanargmax(f1))

    return (
        float(thresholds[idx]),
        float(p[idx]),
        float(r[idx]),
        float(f1[idx]),
    )


def topk_metrics(y_true, prob, frac):
    n = len(y_true)
    k = max(1, int(np.ceil(n * frac)))

    order = np.argsort(-prob)
    selected = np.asarray(y_true)[order[:k]]

    tp = int(selected.sum())
    precision = tp / k
    recall = tp / int(np.asarray(y_true).sum())

    base_rate = np.asarray(y_true).mean()
    lift = precision / base_rate

    return (
        k,
        tp,
        precision,
        recall,
        lift,
    )


def evaluate(
    model,
    candidate,
    y_true,
    prob,
    seconds,
    params_text
):
    roc = roc_auc_score(y_true, prob)
    pr = average_precision_score(y_true, prob)

    threshold, p, r, f1 = best_f1_info(
        y_true,
        prob
    )

    _, top5_tp, top5_p, top5_r, top5_lift = topk_metrics(
        y_true, prob, 0.05
    )

    _, top10_tp, top10_p, top10_r, top10_lift = topk_metrics(
        y_true, prob, 0.10
    )

    _, top20_tp, top20_p, top20_r, top20_lift = topk_metrics(
        y_true, prob, 0.20
    )

    return {
        "Model": model,
        "Candidate": candidate,
        "ROC_AUC": roc,
        "PR_AUC": pr,

        "Best_F1_Threshold": threshold,
        "Precision@BestF1": p,
        "Recall@BestF1": r,
        "Best_F1": f1,

        "Top5_TP": top5_tp,
        "Top5_Precision": top5_p,
        "Top5_Recall": top5_r,
        "Top5_Lift": top5_lift,

        "Top10_TP": top10_tp,
        "Top10_Precision": top10_p,
        "Top10_Recall": top10_r,
        "Top10_Lift": top10_lift,

        "Top20_TP": top20_tp,
        "Top20_Precision": top20_p,
        "Top20_Recall": top20_r,
        "Top20_Lift": top20_lift,

        "Train_Seconds": seconds,
        "Params": params_text,
    }


def print_row(row):
    print(
        f"\n[{row['Candidate']}]"
        f"\n  ROC-AUC = {row['ROC_AUC']:.6f}"
        f"\n  PR-AUC  = {row['PR_AUC']:.6f}"
        f"\n  BestF1  = {row['Best_F1']:.6f} "
        f"(threshold={row['Best_F1_Threshold']:.6f})"
        f"\n  Top10   = TP {row['Top10_TP']} | "
        f"Recall {row['Top10_Recall']:.4f} | "
        f"Lift {row['Top10_Lift']:.2f}x"
        f"\n  Time    = {row['Train_Seconds']:.3f}s"
    )


# ============================================================
# 4. 数据
# ============================================================

print("=" * 84)
print("银行贷款审批预测：模型参数调优 V1.0")
print("=" * 84)

train = load_csv(TRAIN_PATH)
valid = load_csv(VALID_PATH)
test = load_csv(TEST_PATH)

print(f"Train: {train.shape}")
print(f"Valid: {valid.shape}")
print(f"Test : {test.shape}")
print(f"RANDOM_STATE = {RANDOM_STATE}")

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

valid_cols = [
    c for c in valid.columns
    if c != TARGET
]

test_cols = [
    c for c in test.columns
    if c != ID_COL
]

if feature_cols != valid_cols:
    raise ValueError("Train / Valid 特征列不一致")
if feature_cols != test_cols:
    raise ValueError("Train / Test 特征列不一致")

X_train = train[feature_cols].copy()
y_train = train[TARGET].astype(int).copy()

X_valid = valid[feature_cols].copy()
y_valid = valid[TARGET].astype(int).copy()

numeric_features = [
    c for c in feature_cols
    if c not in CATEGORICAL_FEATURES
]

print(
    f"\nTrain positive rate = "
    f"{y_train.mean() * 100:.6f}%"
)

print(
    f"Valid positive rate = "
    f"{y_valid.mean() * 100:.6f}%"
)


# ============================================================
# 5. 结果容器
# ============================================================

results = []
predictions = pd.DataFrame({
    "y_true": y_valid.values
})


# ============================================================
# 6. Logistic Regression
# ============================================================

print("\n" + "=" * 84)
print("A. Logistic Regression：C 调优")
print("=" * 84)

lr_preprocessor = ColumnTransformer(
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

for c_value in LR_C_VALUES:

    candidate = f"LR_C={c_value}"

    model = Pipeline(
        steps=[
            ("preprocess", lr_preprocessor),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    solver="liblinear",
                    class_weight=None,
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )

    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start

    prob = model.predict_proba(X_valid)[:, 1]

    row = evaluate(
        "Logistic Regression",
        candidate,
        y_valid,
        prob,
        seconds,
        f"C={c_value}, class_weight=None"
    )

    results.append(row)
    predictions[candidate] = prob

    print_row(row)


# ============================================================
# 7. CatBoost
# ============================================================

print("\n" + "=" * 84)
print("B. CatBoost：无权重参数调优")
print("=" * 84)

X_train_cat = X_train.copy()
X_valid_cat = X_valid.copy()

for col in CATEGORICAL_FEATURES:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_valid_cat[col] = X_valid_cat[col].astype(str)

for cfg in CAT_CONFIGS:

    candidate = cfg["name"]

    model = CatBoostClassifier(
        iterations=cfg["iterations"],
        depth=cfg["depth"],
        learning_rate=cfg["learning_rate"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False
    )

    start = time.time()

    model.fit(
        X_train_cat,
        y_train,
        cat_features=CATEGORICAL_FEATURES
    )

    seconds = time.time() - start

    prob = model.predict_proba(
        X_valid_cat
    )[:, 1]

    params_text = (
        f"iterations={cfg['iterations']}, "
        f"depth={cfg['depth']}, "
        f"learning_rate={cfg['learning_rate']}, "
        f"l2_leaf_reg={cfg['l2_leaf_reg']}, "
        f"class_weight=None"
    )

    row = evaluate(
        "CatBoost",
        candidate,
        y_valid,
        prob,
        seconds,
        params_text
    )

    results.append(row)
    predictions[candidate] = prob

    print_row(row)


# ============================================================
# 8. XGBoost
# ============================================================

print("\n" + "=" * 84)
print("C. XGBoost：固定 scale_pos_weight=20 后调优")
print("=" * 84)

for cfg in XGB_CONFIGS:

    candidate = cfg["name"]

    model = XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        min_child_weight=cfg["min_child_weight"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        gamma=cfg["gamma"],
        reg_lambda=cfg["reg_lambda"],
        reg_alpha=cfg["reg_alpha"],

        scale_pos_weight=20,

        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start

    prob = model.predict_proba(
        X_valid
    )[:, 1]

    params_text = (
        f"n_estimators={cfg['n_estimators']}, "
        f"max_depth={cfg['max_depth']}, "
        f"learning_rate={cfg['learning_rate']}, "
        f"min_child_weight={cfg['min_child_weight']}, "
        f"subsample={cfg['subsample']}, "
        f"colsample_bytree={cfg['colsample_bytree']}, "
        f"gamma={cfg['gamma']}, "
        f"reg_lambda={cfg['reg_lambda']}, "
        f"reg_alpha={cfg['reg_alpha']}, "
        f"scale_pos_weight=20"
    )

    row = evaluate(
        "XGBoost",
        candidate,
        y_valid,
        prob,
        seconds,
        params_text
    )

    results.append(row)
    predictions[candidate] = prob

    print_row(row)


# ============================================================
# 9. 汇总
# ============================================================

results_df = pd.DataFrame(results)

results_df = (
    results_df
    .sort_values(
        ["ROC_AUC", "PR_AUC"],
        ascending=False
    )
    .reset_index(drop=True)
)

best_by_model = (
    results_df
    .sort_values(
        ["Model", "ROC_AUC", "PR_AUC"],
        ascending=[True, False, False]
    )
    .groupby(
        "Model",
        as_index=False
    )
    .first()
)

print("\n" + "=" * 84)
print("全部候选参数结果（按 ROC-AUC 排序）")
print("=" * 84)

print(
    results_df[
        [
            "Model",
            "Candidate",
            "ROC_AUC",
            "PR_AUC",
            "Best_F1",
            "Top10_Recall",
            "Top10_Lift",
            "Train_Seconds",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 84)
print("每个模型最佳参数")
print("=" * 84)

print(
    best_by_model[
        [
            "Model",
            "Candidate",
            "ROC_AUC",
            "PR_AUC",
            "Best_F1",
            "Top10_Recall",
            "Top10_Lift",
            "Params",
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. 保存
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "tuning_results.csv"
)

best_path = os.path.join(
    OUTPUT_DIR,
    "tuning_best_by_model.csv"
)

pred_path = os.path.join(
    OUTPUT_DIR,
    "tuning_valid_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)

best_by_model.to_csv(
    best_path,
    index=False,
    encoding="utf-8-sig"
)

predictions.to_csv(
    pred_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ 已保存：{results_path}")
print(f"✅ 已保存：{best_path}")
print(f"✅ 已保存：{pred_path}")


# ============================================================
# 11. 摘要
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "tuning_summary.txt"
)

best_overall = results_df.iloc[0]

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
        "银行贷款审批预测 - 模型参数调优 V1.0\n"
    )
    f.write("=" * 76 + "\n")

    f.write(
        f"RANDOM_STATE = {RANDOM_STATE}\n"
    )

    f.write(
        f"Train = {X_train.shape}, "
        f"positive rate={y_train.mean():.8f}\n"
    )

    f.write(
        f"Valid = {X_valid.shape}, "
        f"positive rate={y_valid.mean():.8f}\n"
    )

    f.write(
        "\n调优原则：\n"
        "- Logistic：无 class_weight，只调 C\n"
        "- CatBoost：无 class weight，调 iterations/depth/learning_rate/L2\n"
        "- XGBoost：固定 scale_pos_weight=20，调树结构与正则\n"
        "- Test 未参与模型选择\n"
    )

    f.write(
        "\nROC-AUC 全局最佳："
        f"{best_overall['Model']} | "
        f"{best_overall['Candidate']} | "
        f"{best_overall['ROC_AUC']:.6f}\n"
    )

    f.write(
        "PR-AUC 全局最佳："
        f"{best_pr['Model']} | "
        f"{best_pr['Candidate']} | "
        f"{best_pr['PR_AUC']:.6f}\n"
    )

    f.write(
        "\n每个模型最佳：\n"
    )

    f.write(
        best_by_model[
            [
                "Model",
                "Candidate",
                "ROC_AUC",
                "PR_AUC",
                "Best_F1",
                "Top10_Recall",
                "Top10_Lift",
                "Params",
            ]
        ].to_string(index=False)
    )

    f.write(
        "\n\n说明：\n"
        "1. ROC-AUC 为项目主排名指标。\n"
        "2. PR-AUC 对极度不平衡数据具有重要参考价值。\n"
        "3. 本轮是小规模、预先定义候选参数搜索，不是大规模暴力网格搜索。\n"
        "4. Best-F1 阈值不是最终业务阈值。\n"
        "5. 下一阶段应进行特征消融/时间稳定性检查，再决定最终模型与是否融合。\n"
    )

print(f"✅ 已保存：{summary_path}")

print("\n" + "=" * 84)
print("模型参数调优 V1.0 完成")
print("=" * 84)

print(
    "\n请把 tuning_v1 文件夹中的以下文件发给我："
)
print("1. tuning_results.csv")
print("2. tuning_best_by_model.csv")
print("3. tuning_valid_predictions.csv")
print("4. tuning_summary.txt")
