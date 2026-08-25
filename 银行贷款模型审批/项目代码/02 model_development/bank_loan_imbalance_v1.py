# -*- coding: utf-8 -*-
"""
银行贷款审批预测：类别不平衡处理实验 V1.0
======================================

基于：
- train_processed_v14.csv
- valid_processed_v14.csv
- test_processed_v14.csv

固定随机种子：
RANDOM_STATE = 42

本阶段目标：
1. 不改 V1.4 数据预处理；
2. 不做系统调参；
3. 仅针对 Baseline 表现最好的三个模型进行“类别不平衡策略”对比：
   - Logistic Regression
   - XGBoost
   - CatBoost
4. 统一比较：
   - ROC-AUC
   - PR-AUC
   - 默认 0.5 阈值指标
   - Validation 上探索性 Best-F1 阈值
   - Top 5% / 10% / 20% 人群命中效果
5. Test 数据只做列结构检查，不参与策略选择。

重要说明：
- 本轮不做普通 SMOTE。
  原因：当前含多列类别编码特征，直接使用普通 SMOTE 可能生成不合法的“类别插值”。
- Best F1 threshold 仅用于探索，不是最终业务阈值。
- 不平衡权重提升 Recall 的同时，可能损害概率校准，因此本轮以 ROC-AUC / PR-AUC 为主要选型依据。
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
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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
# 1. 配置区
# ============================================================

DATA_DIR = r"D:\深圳点宽\银行贷款审批\output_v14"

TRAIN_PATH = os.path.join(DATA_DIR, "train_processed_v14.csv")
VALID_PATH = os.path.join(DATA_DIR, "valid_processed_v14.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_processed_v14.csv")

OUTPUT_DIR = os.path.join(DATA_DIR, "imbalance_v1")
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

# XGBoost / CatBoost 需要测试的正类权重
# 最后一个会自动替换为训练集负/正样本比
MANUAL_WEIGHTS = [5, 10, 20, 40]


# ============================================================
# 2. 工具函数
# ============================================================

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    return pd.read_csv(path)


def calc_best_f1_threshold(y_true, prob):
    precision, recall, thresholds = precision_recall_curve(y_true, prob)

    if len(thresholds) == 0:
        return {
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

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


def topk_metrics(y_true, prob, frac):
    """
    只跟进预测概率最高的 frac 人群时：
    - Precision = 该人群真实正样本占比
    - Recall = 捕获全部真实正样本的比例
    - Lift = Precision / 全体验证集正样本率
    """
    n = len(y_true)
    k = max(1, int(np.ceil(n * frac)))

    order = np.argsort(-prob)
    idx = order[:k]

    y_arr = np.asarray(y_true)
    selected = y_arr[idx]

    tp = int(selected.sum())
    precision = tp / k
    total_pos = int(y_arr.sum())
    recall = tp / total_pos if total_pos > 0 else 0.0

    base_rate = y_arr.mean()
    lift = precision / base_rate if base_rate > 0 else np.nan

    return {
        "k": k,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def evaluate(model_name, strategy, y_true, prob, train_seconds):
    roc_auc = roc_auc_score(y_true, prob)
    pr_auc = average_precision_score(y_true, prob)

    # 默认阈值 0.5
    pred05 = (prob >= 0.5).astype(int)

    p05 = precision_score(y_true, pred05, zero_division=0)
    r05 = recall_score(y_true, pred05, zero_division=0)
    f105 = f1_score(y_true, pred05, zero_division=0)

    cm05 = confusion_matrix(y_true, pred05)
    tn05, fp05, fn05, tp05 = cm05.ravel()

    # Best F1 threshold
    best = calc_best_f1_threshold(y_true, prob)
    pred_best = (prob >= best["threshold"]).astype(int)

    cmb = confusion_matrix(y_true, pred_best)
    tnb, fpb, fnb, tpb = cmb.ravel()

    # Top-K
    top5 = topk_metrics(y_true, prob, 0.05)
    top10 = topk_metrics(y_true, prob, 0.10)
    top20 = topk_metrics(y_true, prob, 0.20)

    return {
        "Model": model_name,
        "Strategy": strategy,

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

        "Top5_N": top5["k"],
        "Top5_TP": top5["tp"],
        "Top5_Precision": top5["precision"],
        "Top5_Recall": top5["recall"],
        "Top5_Lift": top5["lift"],

        "Top10_N": top10["k"],
        "Top10_TP": top10["tp"],
        "Top10_Precision": top10["precision"],
        "Top10_Recall": top10["recall"],
        "Top10_Lift": top10["lift"],

        "Top20_N": top20["k"],
        "Top20_TP": top20["tp"],
        "Top20_Precision": top20["precision"],
        "Top20_Recall": top20["recall"],
        "Top20_Lift": top20["lift"],

        "Train_Seconds": train_seconds,
    }


def print_result(row):
    print(
        f"\n[{row['Model']} | {row['Strategy']}]"
        f"\n  ROC-AUC = {row['ROC_AUC']:.6f}"
        f"\n  PR-AUC  = {row['PR_AUC']:.6f}"
        f"\n  @0.5    = P {row['Precision@0.5']:.4f} | "
        f"R {row['Recall@0.5']:.4f} | F1 {row['F1@0.5']:.4f}"
        f"\n  BestF1  = threshold {row['Best_F1_Threshold']:.6f} | "
        f"P {row['Precision@BestF1']:.4f} | "
        f"R {row['Recall@BestF1']:.4f} | "
        f"F1 {row['Best_F1']:.4f}"
        f"\n  Top10%  = TP {row['Top10_TP']} | "
        f"Precision {row['Top10_Precision']:.4f} | "
        f"Recall {row['Top10_Recall']:.4f} | "
        f"Lift {row['Top10_Lift']:.2f}x"
        f"\n  Time    = {row['Train_Seconds']:.3f}s"
    )


# ============================================================
# 3. 加载与检查数据
# ============================================================

print("=" * 82)
print("银行贷款审批预测：类别不平衡处理实验 V1.0")
print("=" * 82)

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

feature_cols = [c for c in train.columns if c != TARGET]
valid_feature_cols = [c for c in valid.columns if c != TARGET]
test_feature_cols = [c for c in test.columns if c != ID_COL]

if feature_cols != valid_feature_cols:
    raise ValueError("Train / Valid 特征列不一致")
if feature_cols != test_feature_cols:
    raise ValueError("Train / Test 特征列不一致")

missing_cat_cols = [
    c for c in CATEGORICAL_FEATURES
    if c not in feature_cols
]
if missing_cat_cols:
    raise ValueError(f"缺少类别特征：{missing_cat_cols}")

numeric_features = [
    c for c in feature_cols
    if c not in CATEGORICAL_FEATURES
]

X_train = train[feature_cols].copy()
y_train = train[TARGET].astype(int).copy()

X_valid = valid[feature_cols].copy()
y_valid = valid[TARGET].astype(int).copy()

X_test = test[feature_cols].copy()

if (
    X_train.isna().sum().sum() != 0
    or X_valid.isna().sum().sum() != 0
    or X_test.isna().sum().sum() != 0
):
    raise ValueError("发现 NaN，请先检查 V1.4 预处理结果")

pos = int(y_train.sum())
neg = int((1 - y_train).sum())
imbalance_ratio = neg / pos

print(
    f"\n训练集：正样本 {pos}，负样本 {neg}，"
    f"负/正样本比 = {imbalance_ratio:.4f}"
)
print(
    f"验证集：正样本 {int(y_valid.sum())} / {len(y_valid)}，"
    f"正样本率 = {y_valid.mean() * 100:.6f}%"
)
print(f"固定 RANDOM_STATE = {RANDOM_STATE}")

weights_to_test = MANUAL_WEIGHTS + [round(imbalance_ratio, 4)]


# ============================================================
# 4. 结果容器
# ============================================================

results = []
predictions = pd.DataFrame({
    "y_true": y_valid.values
})


# ============================================================
# 5. Logistic Regression
# ============================================================

print("\n" + "=" * 82)
print("A. Logistic Regression")
print("=" * 82)

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

for class_weight in [None, "balanced"]:

    strategy = (
        "Baseline"
        if class_weight is None
        else "class_weight=balanced"
    )

    model = Pipeline(
        steps=[
            ("preprocess", logistic_preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight=class_weight,
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    prob = model.predict_proba(X_valid)[:, 1]

    row = evaluate(
        "Logistic Regression",
        strategy,
        y_valid,
        prob,
        elapsed
    )

    results.append(row)

    pred_col = (
        "LR_Baseline"
        if class_weight is None
        else "LR_Balanced"
    )
    predictions[pred_col] = prob

    print_result(row)


# ============================================================
# 6. XGBoost
# ============================================================

print("\n" + "=" * 82)
print("B. XGBoost")
print("=" * 82)

xgb_strategies = [("Baseline", 1.0)]

for w in weights_to_test:
    xgb_strategies.append(
        (f"scale_pos_weight={w}", float(w))
    )

for strategy, spw in xgb_strategies:

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        eval_metric="logloss",
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    prob = model.predict_proba(X_valid)[:, 1]

    row = evaluate(
        "XGBoost",
        strategy,
        y_valid,
        prob,
        elapsed
    )
    results.append(row)

    pred_col = (
        "XGB_Baseline"
        if spw == 1.0
        else f"XGB_SPW_{str(spw).replace('.', '_')}"
    )
    predictions[pred_col] = prob

    print_result(row)


# ============================================================
# 7. CatBoost
# ============================================================

print("\n" + "=" * 82)
print("C. CatBoost")
print("=" * 82)

X_train_cat = X_train.copy()
X_valid_cat = X_valid.copy()

for col in CATEGORICAL_FEATURES:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_valid_cat[col] = X_valid_cat[col].astype(str)


# 7.1 Baseline
cat_baseline = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=False
)

start = time.time()
cat_baseline.fit(
    X_train_cat,
    y_train,
    cat_features=CATEGORICAL_FEATURES
)
elapsed = time.time() - start

prob = cat_baseline.predict_proba(X_valid_cat)[:, 1]

row = evaluate(
    "CatBoost",
    "Baseline",
    y_valid,
    prob,
    elapsed
)
results.append(row)
predictions["CAT_Baseline"] = prob
print_result(row)


# 7.2 auto_class_weights='Balanced'
cat_balanced = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    verbose=False
)

start = time.time()
cat_balanced.fit(
    X_train_cat,
    y_train,
    cat_features=CATEGORICAL_FEATURES
)
elapsed = time.time() - start

prob = cat_balanced.predict_proba(X_valid_cat)[:, 1]

row = evaluate(
    "CatBoost",
    "auto_class_weights=Balanced",
    y_valid,
    prob,
    elapsed
)
results.append(row)
predictions["CAT_AutoBalanced"] = prob
print_result(row)


# 7.3 scale_pos_weight
for w in weights_to_test:

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=float(w),
        random_seed=RANDOM_STATE,
        verbose=False
    )

    start = time.time()
    model.fit(
        X_train_cat,
        y_train,
        cat_features=CATEGORICAL_FEATURES
    )
    elapsed = time.time() - start

    prob = model.predict_proba(X_valid_cat)[:, 1]

    row = evaluate(
        "CatBoost",
        f"scale_pos_weight={w}",
        y_valid,
        prob,
        elapsed
    )
    results.append(row)

    pred_col = f"CAT_SPW_{str(w).replace('.', '_')}"
    predictions[pred_col] = prob

    print_result(row)


# ============================================================
# 8. 汇总
# ============================================================

results_df = pd.DataFrame(results)

# 主排序：ROC-AUC，其次 PR-AUC
results_df = (
    results_df
    .sort_values(
        ["ROC_AUC", "PR_AUC"],
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 82)
print("不平衡策略总表（按 ROC-AUC 排序）")
print("=" * 82)

display_cols = [
    "Model",
    "Strategy",
    "ROC_AUC",
    "PR_AUC",
    "Best_F1_Threshold",
    "Precision@BestF1",
    "Recall@BestF1",
    "Best_F1",
    "Top5_Precision",
    "Top5_Recall",
    "Top5_Lift",
    "Top10_Precision",
    "Top10_Recall",
    "Top10_Lift",
    "Top20_Precision",
    "Top20_Recall",
    "Top20_Lift",
    "Train_Seconds",
]

print(
    results_df[display_cols].to_string(index=False)
)


# ============================================================
# 9. 每个模型内部最佳策略
# ============================================================

best_by_model = (
    results_df
    .sort_values(
        ["Model", "ROC_AUC", "PR_AUC"],
        ascending=[True, False, False]
    )
    .groupby("Model", as_index=False)
    .first()
)

print("\n" + "=" * 82)
print("每个模型 ROC-AUC 最佳不平衡策略")
print("=" * 82)

print(
    best_by_model[
        [
            "Model",
            "Strategy",
            "ROC_AUC",
            "PR_AUC",
            "Best_F1_Threshold",
            "Best_F1",
            "Top10_Recall",
            "Top10_Lift",
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. 保存
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "imbalance_results.csv"
)

pred_path = os.path.join(
    OUTPUT_DIR,
    "imbalance_valid_predictions.csv"
)

best_path = os.path.join(
    OUTPUT_DIR,
    "imbalance_best_by_model.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)

predictions.to_csv(
    pred_path,
    index=False,
    encoding="utf-8-sig"
)

best_by_model.to_csv(
    best_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ 已保存：{results_path}")
print(f"✅ 已保存：{pred_path}")
print(f"✅ 已保存：{best_path}")


# ============================================================
# 11. 摘要
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "imbalance_summary.txt"
)

best_overall_roc = results_df.iloc[0]

best_overall_pr = (
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
        "银行贷款审批预测 - 类别不平衡处理实验 V1.0\n"
    )
    f.write("=" * 72 + "\n")

    f.write(
        f"RANDOM_STATE = {RANDOM_STATE}\n"
    )

    f.write(
        f"Train = {X_train.shape}, "
        f"正样本={pos}, 负样本={neg}, "
        f"负/正比={imbalance_ratio:.6f}\n"
    )

    f.write(
        f"Valid = {X_valid.shape}, "
        f"正样本率={y_valid.mean():.8f}\n"
    )

    f.write(
        "\n本轮策略：\n"
        "- Logistic Regression: Baseline / class_weight=balanced\n"
        "- XGBoost: Baseline / scale_pos_weight 5,10,20,40,负正样本比\n"
        "- CatBoost: Baseline / auto_class_weights=Balanced / "
        "scale_pos_weight 5,10,20,40,负正样本比\n"
        "- 未使用普通 SMOTE\n"
    )

    f.write(
        "\nROC-AUC 最佳："
        f"{best_overall_roc['Model']} | "
        f"{best_overall_roc['Strategy']} | "
        f"{best_overall_roc['ROC_AUC']:.6f}\n"
    )

    f.write(
        "PR-AUC 最佳："
        f"{best_overall_pr['Model']} | "
        f"{best_overall_pr['Strategy']} | "
        f"{best_overall_pr['PR_AUC']:.6f}\n"
    )

    f.write(
        "\n每个模型最佳策略：\n"
    )

    f.write(
        best_by_model[
            [
                "Model",
                "Strategy",
                "ROC_AUC",
                "PR_AUC",
                "Best_F1_Threshold",
                "Best_F1",
                "Top10_Recall",
                "Top10_Lift",
            ]
        ].to_string(index=False)
    )

    f.write(
        "\n\n说明：\n"
        "1. 以 ROC-AUC 为主选型指标，PR-AUC 为重要参考。\n"
        "2. Best F1 threshold 仅为 Validation 探索结果。\n"
        "3. Test 未参与任何策略选择。\n"
        "4. 类别权重可能改变预测概率的绝对数值，因此后续最终模型还需单独做阈值与校准分析。\n"
    )

print(f"✅ 已保存：{summary_path}")

print("\n" + "=" * 82)
print("类别不平衡处理实验 V1.0 完成")
print("=" * 82)

print(
    "\n请把 imbalance_v1 文件夹中的以下文件发给我："
)
print("1. imbalance_results.csv")
print("2. imbalance_best_by_model.csv")
print("3. imbalance_valid_predictions.csv")
print("4. imbalance_summary.txt")
