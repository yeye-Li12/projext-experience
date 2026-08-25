# -*- coding: utf-8 -*-
"""
银行贷款审批预测：固定模型融合验证 V1.0
====================================

承接 ablation_v1 的正式结论：

主模型：
1. CatBoost CAT_C08 + FULL 22 features
   iterations=600
   depth=5
   learning_rate=0.05
   l2_leaf_reg=10
   no class weight

2. XGBoost XGB_X08 + DROP_CITY
   删除：
   - City_Code
   - City_Category

   n_estimators=800
   max_depth=3
   learning_rate=0.025
   min_child_weight=10
   subsample=0.9
   colsample_bytree=0.9
   gamma=0.1
   reg_lambda=10
   reg_alpha=0
   scale_pos_weight=20

固定随机种子：
RANDOM_STATE = 42

融合方式：
- 不再搜索融合权重；
- 固定使用：
    60% CatBoost rank score
    40% XGBoost rank score
- 原因：
  CatBoost 是当前 ROC-AUC 主模型；
  XGBoost DROP_CITY 在 PR-AUC 和少数类前端排序上提供补充；
  rank blending 可减弱两个模型概率尺度不同（尤其 XGBoost 加权训练）的影响。

本脚本目的：
1. 重现两个最终候选单模型；
2. 验证固定 60/40 Rank Fusion；
3. 检查 Valid 前半/后半稳定性；
4. 用分层 Bootstrap 比较 Fusion 与 CatBoost 主模型的差异；
5. Test 不参与模型选择。

注意：
- Rank Fusion 输出的是 ranking score，不在本阶段将其解释成“校准后的真实概率”。
- 最终业务概率/业务阈值会在后续概率校准与阈值阶段单独处理。
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

try:
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError(
        "缺少依赖，请先执行：\n"
        "pip install xgboost catboost pandas numpy scikit-learn\n"
        f"\n原始错误：{e}"
    )


# ============================================================
# 1. 配置
# ============================================================

DATA_DIR = r"D:\深圳点宽\银行贷款审批\output_v14"

TRAIN_PATH = os.path.join(DATA_DIR, "train_processed_v14.csv")
VALID_PATH = os.path.join(DATA_DIR, "valid_processed_v14.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_processed_v14.csv")

OUTPUT_DIR = os.path.join(DATA_DIR, "fusion_v1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

TARGET = "Approved"
ID_COL = "ID"

CAT_WEIGHT = 0.60
XGB_WEIGHT = 0.40

BOOTSTRAP_ROUNDS = 2000

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

XGB_DROP_FEATURES = [
    "City_Code",
    "City_Category",
]


# ============================================================
# 2. 工具函数
# ============================================================

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    return pd.read_csv(path)


def rank_score(values):
    """
    转成 0~1 百分位排名。
    ROC-AUC 只依赖排序，因此 rank transform 不改变单模型 ROC-AUC。
    """
    return (
        pd.Series(values)
        .rank(method="average", pct=True)
        .to_numpy()
    )


def calc_metrics(y_true, score):
    return {
        "ROC_AUC": roc_auc_score(y_true, score),
        "PR_AUC": average_precision_score(y_true, score),
    }


def topk_metrics(y_true, score, frac):
    y = np.asarray(y_true)
    score = np.asarray(score)

    k = max(1, int(np.ceil(len(y) * frac)))
    idx = np.argsort(-score)[:k]

    tp = int(y[idx].sum())
    precision = tp / k
    recall = tp / int(y.sum())

    base_rate = y.mean()
    lift = precision / base_rate

    return {
        "N": k,
        "TP": tp,
        "Precision": precision,
        "Recall": recall,
        "Lift": lift,
    }


def evaluate(name, y_true, score):
    m = calc_metrics(y_true, score)

    t5 = topk_metrics(y_true, score, 0.05)
    t10 = topk_metrics(y_true, score, 0.10)
    t20 = topk_metrics(y_true, score, 0.20)

    return {
        "Model": name,
        "ROC_AUC": m["ROC_AUC"],
        "PR_AUC": m["PR_AUC"],

        "Top5_TP": t5["TP"],
        "Top5_Precision": t5["Precision"],
        "Top5_Recall": t5["Recall"],
        "Top5_Lift": t5["Lift"],

        "Top10_TP": t10["TP"],
        "Top10_Precision": t10["Precision"],
        "Top10_Recall": t10["Recall"],
        "Top10_Lift": t10["Lift"],

        "Top20_TP": t20["TP"],
        "Top20_Precision": t20["Precision"],
        "Top20_Recall": t20["Recall"],
        "Top20_Lift": t20["Lift"],
    }


def half_metrics(y_true, score):
    y = np.asarray(y_true)
    score = np.asarray(score)

    mid = len(y) // 2

    early = calc_metrics(
        y[:mid],
        score[:mid]
    )

    late = calc_metrics(
        y[mid:],
        score[mid:]
    )

    return {
        "Early_ROC_AUC": early["ROC_AUC"],
        "Early_PR_AUC": early["PR_AUC"],
        "Late_ROC_AUC": late["ROC_AUC"],
        "Late_PR_AUC": late["PR_AUC"],
        "ROC_Stability_Gap": abs(
            late["ROC_AUC"] - early["ROC_AUC"]
        ),
        "PR_Stability_Gap": abs(
            late["PR_AUC"] - early["PR_AUC"]
        ),
    }


def stratified_bootstrap_compare(
    y_true,
    reference_score,
    candidate_score,
    rounds=2000,
    seed=42
):
    """
    分层 Bootstrap：
    分别对正样本和负样本有放回抽样，
    比较 candidate - reference 的 ROC-AUC / PR-AUC 差异。
    """
    y = np.asarray(y_true)
    ref = np.asarray(reference_score)
    cand = np.asarray(candidate_score)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng = np.random.default_rng(seed)

    rows = []

    for i in range(rounds):

        sampled_pos = rng.choice(
            pos_idx,
            size=len(pos_idx),
            replace=True
        )

        sampled_neg = rng.choice(
            neg_idx,
            size=len(neg_idx),
            replace=True
        )

        idx = np.concatenate([
            sampled_pos,
            sampled_neg
        ])

        yy = y[idx]

        ref_auc = roc_auc_score(
            yy,
            ref[idx]
        )

        cand_auc = roc_auc_score(
            yy,
            cand[idx]
        )

        ref_pr = average_precision_score(
            yy,
            ref[idx]
        )

        cand_pr = average_precision_score(
            yy,
            cand[idx]
        )

        rows.append({
            "round": i + 1,
            "ROC_Delta_Fusion_minus_Cat": (
                cand_auc - ref_auc
            ),
            "PR_Delta_Fusion_minus_Cat": (
                cand_pr - ref_pr
            ),
        })

    boot = pd.DataFrame(rows)

    roc_arr = boot[
        "ROC_Delta_Fusion_minus_Cat"
    ].to_numpy()

    pr_arr = boot[
        "PR_Delta_Fusion_minus_Cat"
    ].to_numpy()

    summary = {
        "Bootstrap_Rounds": rounds,

        "ROC_Delta_Mean": float(
            roc_arr.mean()
        ),
        "ROC_Delta_CI_Low": float(
            np.quantile(roc_arr, 0.025)
        ),
        "ROC_Delta_Median": float(
            np.quantile(roc_arr, 0.50)
        ),
        "ROC_Delta_CI_High": float(
            np.quantile(roc_arr, 0.975)
        ),
        "ROC_Pct_Delta_GT_0": float(
            (roc_arr > 0).mean()
        ),

        "PR_Delta_Mean": float(
            pr_arr.mean()
        ),
        "PR_Delta_CI_Low": float(
            np.quantile(pr_arr, 0.025)
        ),
        "PR_Delta_Median": float(
            np.quantile(pr_arr, 0.50)
        ),
        "PR_Delta_CI_High": float(
            np.quantile(pr_arr, 0.975)
        ),
        "PR_Pct_Delta_GT_0": float(
            (pr_arr > 0).mean()
        ),
    }

    return boot, summary


# ============================================================
# 3. 读取数据
# ============================================================

print("=" * 86)
print("银行贷款审批预测：固定模型融合验证 V1.0")
print("=" * 86)

train = load_csv(TRAIN_PATH)
valid = load_csv(VALID_PATH)
test = load_csv(TEST_PATH)

print(f"Train: {train.shape}")
print(f"Valid: {valid.shape}")
print(f"Test : {test.shape}")
print(f"RANDOM_STATE = {RANDOM_STATE}")

all_features = [
    c for c in train.columns
    if c != TARGET
]

valid_features = [
    c for c in valid.columns
    if c != TARGET
]

test_features = [
    c for c in test.columns
    if c != ID_COL
]

if all_features != valid_features:
    raise ValueError(
        "Train / Valid 特征列不一致"
    )

if all_features != test_features:
    raise ValueError(
        "Train / Test 特征列不一致"
    )

X_train = train[
    all_features
].copy()

X_valid = valid[
    all_features
].copy()

y_train = train[
    TARGET
].astype(int).copy()

y_valid = valid[
    TARGET
].astype(int).copy()

print(
    f"\nValid 正样本："
    f"{int(y_valid.sum())}/{len(y_valid)} "
    f"({y_valid.mean() * 100:.6f}%)"
)


# ============================================================
# 4. CatBoost FULL
# ============================================================

print("\n" + "=" * 86)
print("A. CatBoost CAT_C08 | FULL")
print("=" * 86)

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
    iterations=600,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=10,
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

cat_seconds = time.time() - start

cat_prob = cat_model.predict_proba(
    X_valid_cat
)[:, 1]

cat_rank = rank_score(
    cat_prob
)

cat_eval = evaluate(
    "CatBoost_CAT_C08_FULL",
    y_valid,
    cat_prob
)

cat_half = half_metrics(
    y_valid,
    cat_prob
)

print(
    f"ROC-AUC={cat_eval['ROC_AUC']:.6f} | "
    f"PR-AUC={cat_eval['PR_AUC']:.6f} | "
    f"Top10 Recall={cat_eval['Top10_Recall']:.4f}"
)


# ============================================================
# 5. XGBoost DROP_CITY
# ============================================================

print("\n" + "=" * 86)
print("B. XGBoost XGB_X08 | DROP_CITY")
print("=" * 86)

xgb_features = [
    c for c in all_features
    if c not in XGB_DROP_FEATURES
]

X_train_xgb = train[
    xgb_features
].copy()

X_valid_xgb = valid[
    xgb_features
].copy()

xgb_model = XGBClassifier(
    n_estimators=800,
    max_depth=3,
    learning_rate=0.025,
    min_child_weight=10,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_lambda=10.0,
    reg_alpha=0.0,
    scale_pos_weight=20,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

start = time.time()

xgb_model.fit(
    X_train_xgb,
    y_train
)

xgb_seconds = time.time() - start

xgb_prob = xgb_model.predict_proba(
    X_valid_xgb
)[:, 1]

xgb_rank = rank_score(
    xgb_prob
)

xgb_eval = evaluate(
    "XGBoost_X08_DROP_CITY",
    y_valid,
    xgb_prob
)

xgb_half = half_metrics(
    y_valid,
    xgb_prob
)

print(
    f"ROC-AUC={xgb_eval['ROC_AUC']:.6f} | "
    f"PR-AUC={xgb_eval['PR_AUC']:.6f} | "
    f"Top10 Recall={xgb_eval['Top10_Recall']:.4f}"
)


# ============================================================
# 6. 固定 60/40 Rank Fusion
# ============================================================

print("\n" + "=" * 86)
print("C. Fixed Rank Fusion | CAT 60% + XGB 40%")
print("=" * 86)

fusion_score = (
    CAT_WEIGHT * cat_rank
    + XGB_WEIGHT * xgb_rank
)

fusion_eval = evaluate(
    "RankFusion_CAT60_XGB40",
    y_valid,
    fusion_score
)

fusion_half = half_metrics(
    y_valid,
    fusion_score
)

print(
    f"ROC-AUC={fusion_eval['ROC_AUC']:.6f} | "
    f"PR-AUC={fusion_eval['PR_AUC']:.6f} | "
    f"Top10 Recall={fusion_eval['Top10_Recall']:.4f}"
)

print(
    f"Early ROC={fusion_half['Early_ROC_AUC']:.6f} | "
    f"Late ROC={fusion_half['Late_ROC_AUC']:.6f} | "
    f"Gap={fusion_half['ROC_Stability_Gap']:.6f}"
)


# ============================================================
# 7. 汇总结果
# ============================================================

rows = []

for eval_row, half_row, train_seconds in [
    (
        cat_eval,
        cat_half,
        cat_seconds
    ),
    (
        xgb_eval,
        xgb_half,
        xgb_seconds
    ),
    (
        fusion_eval,
        fusion_half,
        cat_seconds + xgb_seconds
    ),
]:

    row = dict(eval_row)
    row.update(half_row)
    row["Train_Seconds"] = train_seconds

    rows.append(row)

results_df = pd.DataFrame(
    rows
)

results_df = (
    results_df
    .sort_values(
        [
            "ROC_AUC",
            "PR_AUC"
        ],
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 86)
print("融合验证结果")
print("=" * 86)

print(
    results_df[
        [
            "Model",
            "ROC_AUC",
            "PR_AUC",
            "Top5_Recall",
            "Top10_Recall",
            "Top20_Recall",
            "Early_ROC_AUC",
            "Late_ROC_AUC",
            "ROC_Stability_Gap",
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. Bootstrap：Fusion vs CatBoost
# ============================================================

print("\n" + "=" * 86)
print("Bootstrap：Fusion vs CatBoost")
print("=" * 86)

bootstrap_df, bootstrap_summary = (
    stratified_bootstrap_compare(
        y_valid,
        reference_score=cat_prob,
        candidate_score=fusion_score,
        rounds=BOOTSTRAP_ROUNDS,
        seed=RANDOM_STATE
    )
)

print(
    "ROC Δ(Fusion-Cat): "
    f"mean={bootstrap_summary['ROC_Delta_Mean']:.6f}, "
    f"95%CI=["
    f"{bootstrap_summary['ROC_Delta_CI_Low']:.6f}, "
    f"{bootstrap_summary['ROC_Delta_CI_High']:.6f}], "
    f"P(Δ>0)="
    f"{bootstrap_summary['ROC_Pct_Delta_GT_0']:.3f}"
)

print(
    "PR Δ(Fusion-Cat): "
    f"mean={bootstrap_summary['PR_Delta_Mean']:.6f}, "
    f"95%CI=["
    f"{bootstrap_summary['PR_Delta_CI_Low']:.6f}, "
    f"{bootstrap_summary['PR_Delta_CI_High']:.6f}], "
    f"P(Δ>0)="
    f"{bootstrap_summary['PR_Pct_Delta_GT_0']:.3f}"
)


# ============================================================
# 9. 保存
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "fusion_results.csv"
)

pred_path = os.path.join(
    OUTPUT_DIR,
    "fusion_valid_predictions.csv"
)

bootstrap_path = os.path.join(
    OUTPUT_DIR,
    "fusion_bootstrap.csv"
)

bootstrap_summary_path = os.path.join(
    OUTPUT_DIR,
    "fusion_bootstrap_summary.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)

predictions_df = pd.DataFrame({
    "y_true": y_valid.values,
    "CatBoost_Probability": cat_prob,
    "XGBoost_Probability": xgb_prob,
    "CatBoost_Rank": cat_rank,
    "XGBoost_Rank": xgb_rank,
    "Fusion_Rank_Score": fusion_score,
})

predictions_df.to_csv(
    pred_path,
    index=False,
    encoding="utf-8-sig"
)

bootstrap_df.to_csv(
    bootstrap_path,
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([
    bootstrap_summary
]).to_csv(
    bootstrap_summary_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ 已保存：{results_path}")
print(f"✅ 已保存：{pred_path}")
print(f"✅ 已保存：{bootstrap_path}")
print(f"✅ 已保存：{bootstrap_summary_path}")


# ============================================================
# 10. 摘要
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "fusion_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "银行贷款审批预测 - 固定模型融合验证 V1.0\n"
    )
    f.write("=" * 78 + "\n")

    f.write(
        f"RANDOM_STATE = {RANDOM_STATE}\n"
    )

    f.write(
        "固定融合："
        f"CatBoost Rank {CAT_WEIGHT:.0%} + "
        f"XGBoost Rank {XGB_WEIGHT:.0%}\n"
    )

    f.write(
        "\n候选模型：\n"
        "- CatBoost CAT_C08 + FULL\n"
        "- XGBoost XGB_X08 + DROP_CITY\n"
        "- Fixed Rank Fusion 60/40\n"
    )

    f.write(
        "\n结果：\n"
    )

    f.write(
        results_df[
            [
                "Model",
                "ROC_AUC",
                "PR_AUC",
                "Top10_Recall",
                "Top10_Lift",
                "Early_ROC_AUC",
                "Late_ROC_AUC",
                "ROC_Stability_Gap",
            ]
        ].to_string(index=False)
    )

    f.write(
        "\n\nBootstrap Fusion vs CatBoost：\n"
    )

    for k, v in bootstrap_summary.items():
        f.write(
            f"{k} = {v}\n"
        )

    f.write(
        "\n说明：\n"
        "1. 本轮没有搜索融合权重，固定使用 60/40。\n"
        "2. Test 未参与模型选择。\n"
        "3. Rank Fusion 是排序分数，不直接等于校准后的获批概率。\n"
        "4. 如果 Fusion 在 ROC、PR 和时间稳定性上均保持优势，"
        "下一步即可冻结预测方案并进入最终全量训练、概率校准/阈值与 submission 阶段。\n"
    )

print(f"✅ 已保存：{summary_path}")

print("\n" + "=" * 86)
print("固定模型融合验证 V1.0 完成")
print("=" * 86)

print(
    "\n请把 fusion_v1 文件夹中的以下文件发给我："
)
print("1. fusion_results.csv")
print("2. fusion_valid_predictions.csv")
print("3. fusion_bootstrap_summary.csv")
print("4. fusion_summary.txt")
