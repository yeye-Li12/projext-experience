# -*- coding: utf-8 -*-
"""
银行贷款审批预测：LightGBM PR-AUC 导向优化 + 特征消融 V1.0
=========================================================

目标：
1. 为后续 XGBoost + CatBoost + LightGBM OOF Stacking 补强 LightGBM；
2. 主指标改为 PR-AUC，ROC-AUC 为参考；
3. 尽量减少继续对外层 Validation 的反复搜索：
   - 参数/特征集选择主要在 train_processed_v14 内部按时间顺序做 Expanding-Window CV；
   - 只将内部 CV 最优候选在 valid_processed_v14 上做一次外层确认。
4. Test 不参与任何模型选择。

数据：
- train_processed_v14.csv：35184 行，开发训练区
- valid_processed_v14.csv：8796 行，外层时间验证区
- test_processed_v14.csv：仅做结构检查

固定：
RANDOM_STATE = 42

说明：
- 当前 processed 数据来自 V1.4 开发阶段预处理。
- 内部 CV 主要用于减少 LightGBM 候选搜索对外层 Valid 的依赖；
  最终是否进入 Stacking，以外层 Valid PR-AUC + 与其他模型的互补性共同判断。
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError(
        "缺少 LightGBM。请先执行：\n"
        "pip install lightgbm scikit-learn pandas numpy\n"
        f"\n原始错误：{e}"
    )


# ============================================================
# 1. 路径 / 基本配置
# ============================================================

DATA_DIR = r"D:\深圳点宽\银行贷款审批\output_v14"

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train_processed_v14.csv"
)

VALID_PATH = os.path.join(
    DATA_DIR,
    "valid_processed_v14.csv"
)

TEST_PATH = os.path.join(
    DATA_DIR,
    "test_processed_v14.csv"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "lightgbm_v1"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42

TARGET = "Approved"
ID_COL = "ID"

ALL_CATEGORICAL_FEATURES = [
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
# 2. 特征集
# ============================================================

# 控制规模：只测试最有理由的三个版本。
FEATURE_SETS = {
    "FULL": [],

    # XGBoost 消融中表现最好，因此 LightGBM 也测试
    "DROP_CITY": [
        "City_Code",
        "City_Category",
    ],

    # Source 在未来 Test 中分布变化极大，作为稳健性对照
    "DROP_SOURCE": [
        "Source",
        "Source_Category",
    ],
}


# ============================================================
# 3. LightGBM 候选配置
# ============================================================

# 这里不是大网格搜索，而是预先定义的小规模候选。
# scale_pos_weight 重点覆盖 1 / 10 / 20 / 40，
# 同时测试更强正则与更浅树。

LGB_CONFIGS = [
    {
        "name": "LGB_C00_BASE",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": 1.0,
    },

    {
        "name": "LGB_C01_SPW10",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": 10.0,
    },

    {
        "name": "LGB_C02_SPW20",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": 20.0,
    },

    {
        "name": "LGB_C03_SPW40",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": 40.0,
    },

    {
        "name": "LGB_C04_REG15",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 50,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.0,
        "reg_lambda": 5.0,
        "scale_pos_weight": 20.0,
    },

    {
        "name": "LGB_C05_REG31",
        "n_estimators": 700,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 10.0,
        "scale_pos_weight": 20.0,
    },

    {
        "name": "LGB_C06_SHALLOW",
        "n_estimators": 700,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 100,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 10.0,
        "scale_pos_weight": 10.0,
    },

    {
        "name": "LGB_C07_REG63",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 50,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 10.0,
        "scale_pos_weight": 20.0,
    },
]


# ============================================================
# 4. 内部时间 CV 定义
# ============================================================

# train_processed_v14 保留时间排序后的行顺序。
# 使用 expanding-window：
# Fold 1: 前40%训练 -> 接下来15%验证
# Fold 2: 前55%训练 -> 接下来15%验证
# Fold 3: 前70%训练 -> 接下来15%验证
# Fold 4: 前85%训练 -> 最后15%验证

CV_FOLDS = [
    (0.00, 0.40, 0.40, 0.55),
    (0.00, 0.55, 0.55, 0.70),
    (0.00, 0.70, 0.70, 0.85),
    (0.00, 0.85, 0.85, 1.00),
]


# ============================================================
# 5. 工具函数
# ============================================================

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )
    return pd.read_csv(path)


def build_fold_indices(
    n,
    fold_tuple
):
    _, train_end_frac, valid_start_frac, valid_end_frac = fold_tuple

    train_end = int(
        np.floor(
            n * train_end_frac
        )
    )

    valid_start = int(
        np.floor(
            n * valid_start_frac
        )
    )

    valid_end = int(
        np.floor(
            n * valid_end_frac
        )
    )

    train_idx = np.arange(
        0,
        train_end
    )

    valid_idx = np.arange(
        valid_start,
        valid_end
    )

    return train_idx, valid_idx


def make_lgb_frames(
    train_frame,
    valid_frame,
    cat_features
):
    """
    LightGBM 使用 pandas category。

    为保证 Train / Valid category dtype 一致，
    类别集合使用本次 fold 训练部分中已出现的编码值 + [-1]。
    Valid 中不在训练类别集合里的值设为 -1。
    """
    tr = train_frame.copy()
    va = valid_frame.copy()

    for col in cat_features:

        train_values = (
            tr[col]
            .astype(int)
        )

        categories = sorted(
            set(
                train_values.tolist()
            )
            | {-1}
        )

        category_set = set(
            categories
        )

        tr_values = (
            tr[col]
            .astype(int)
            .where(
                tr[col]
                .astype(int)
                .isin(category_set),
                -1
            )
        )

        va_values = (
            va[col]
            .astype(int)
            .where(
                va[col]
                .astype(int)
                .isin(category_set),
                -1
            )
        )

        tr[col] = pd.Categorical(
            tr_values,
            categories=categories
        )

        va[col] = pd.Categorical(
            va_values,
            categories=categories
        )

    return tr, va


def create_model(cfg):
    return LGBMClassifier(
        objective="binary",

        n_estimators=cfg[
            "n_estimators"
        ],

        learning_rate=cfg[
            "learning_rate"
        ],

        num_leaves=cfg[
            "num_leaves"
        ],

        max_depth=cfg[
            "max_depth"
        ],

        min_child_samples=cfg[
            "min_child_samples"
        ],

        subsample=cfg[
            "subsample"
        ],

        colsample_bytree=cfg[
            "colsample_bytree"
        ],

        reg_alpha=cfg[
            "reg_alpha"
        ],

        reg_lambda=cfg[
            "reg_lambda"
        ],

        scale_pos_weight=cfg[
            "scale_pos_weight"
        ],

        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def topk_metrics(
    y_true,
    prob,
    frac=0.10
):
    y = np.asarray(
        y_true
    )

    p = np.asarray(
        prob
    )

    k = max(
        1,
        int(
            np.ceil(
                len(y) * frac
            )
        )
    )

    idx = np.argsort(
        -p
    )[:k]

    tp = int(
        y[idx].sum()
    )

    precision = tp / k

    recall = (
        tp / int(y.sum())
        if y.sum() > 0
        else np.nan
    )

    base_rate = (
        y.mean()
    )

    lift = (
        precision / base_rate
        if base_rate > 0
        else np.nan
    )

    return (
        tp,
        precision,
        recall,
        lift,
    )


# ============================================================
# 6. 加载数据
# ============================================================

print(
    "=" * 90
)

print(
    "LightGBM PR-AUC 导向优化 + 特征消融 V1.0"
)

print(
    "=" * 90
)

train = load_csv(
    TRAIN_PATH
)

valid = load_csv(
    VALID_PATH
)

test = load_csv(
    TEST_PATH
)

print(
    f"Train: {train.shape}"
)

print(
    f"Outer Valid: {valid.shape}"
)

print(
    f"Test: {test.shape}"
)

print(
    f"RANDOM_STATE = "
    f"{RANDOM_STATE}"
)

if TARGET not in train.columns:
    raise ValueError(
        "Train 缺少 Approved"
    )

if TARGET not in valid.columns:
    raise ValueError(
        "Valid 缺少 Approved"
    )

if ID_COL not in test.columns:
    raise ValueError(
        "Test 缺少 ID"
    )

all_features = [
    c
    for c in train.columns
    if c != TARGET
]

valid_features = [
    c
    for c in valid.columns
    if c != TARGET
]

test_features = [
    c
    for c in test.columns
    if c != ID_COL
]

if (
    all_features
    != valid_features
):
    raise ValueError(
        "Train / Valid 特征列不一致"
    )

if (
    all_features
    != test_features
):
    raise ValueError(
        "Train / Test 特征列不一致"
    )

y_train_all = (
    train[TARGET]
    .astype(int)
)

y_valid_outer = (
    valid[TARGET]
    .astype(int)
)

print(
    "\n[标签分布]"
)

print(
    f"Train positives="
    f"{int(y_train_all.sum())}, "
    f"rate="
    f"{y_train_all.mean():.6f}"
)

print(
    f"Outer Valid positives="
    f"{int(y_valid_outer.sum())}, "
    f"rate="
    f"{y_valid_outer.mean():.6f}"
)


# ============================================================
# 7. 内部 Expanding-Window CV
# ============================================================

fold_rows = []

print(
    "\n" + "=" * 90
)

print(
    "A. 内部时间 CV：选择 LightGBM 参数 + 特征集"
)

print(
    "=" * 90
)

for feature_set_name, dropped in FEATURE_SETS.items():

    features = [
        c
        for c in all_features
        if c not in dropped
    ]

    cat_features = [
        c
        for c in ALL_CATEGORICAL_FEATURES
        if c in features
    ]

    print(
        f"\n>>> Feature Set: "
        f"{feature_set_name} "
        f"({len(features)} features)"
    )

    for cfg in LGB_CONFIGS:

        print(
            f"  Candidate "
            f"{cfg['name']}"
        )

        for fold_num, fold_def in enumerate(
            CV_FOLDS,
            start=1
        ):

            train_idx, val_idx = (
                build_fold_indices(
                    len(train),
                    fold_def
                )
            )

            Xtr = (
                train.iloc[
                    train_idx
                ][features]
                .copy()
            )

            ytr = (
                train.iloc[
                    train_idx
                ][TARGET]
                .astype(int)
                .copy()
            )

            Xva = (
                train.iloc[
                    val_idx
                ][features]
                .copy()
            )

            yva = (
                train.iloc[
                    val_idx
                ][TARGET]
                .astype(int)
                .copy()
            )

            Xtr_lgb, Xva_lgb = (
                make_lgb_frames(
                    Xtr,
                    Xva,
                    cat_features
                )
            )

            model = create_model(
                cfg
            )

            start = time.time()

            model.fit(
                Xtr_lgb,
                ytr,
                categorical_feature=cat_features
            )

            seconds = (
                time.time()
                - start
            )

            prob = (
                model
                .predict_proba(
                    Xva_lgb
                )[:, 1]
            )

            pr_auc = (
                average_precision_score(
                    yva,
                    prob
                )
            )

            roc_auc = (
                roc_auc_score(
                    yva,
                    prob
                )
            )

            (
                top10_tp,
                top10_precision,
                top10_recall,
                top10_lift,
            ) = topk_metrics(
                yva,
                prob,
                0.10
            )

            fold_rows.append({
                "Feature_Set":
                    feature_set_name,

                "Candidate":
                    cfg["name"],

                "Fold":
                    fold_num,

                "Train_N":
                    len(train_idx),

                "Train_Positives":
                    int(ytr.sum()),

                "Valid_N":
                    len(val_idx),

                "Valid_Positives":
                    int(yva.sum()),

                "PR_AUC":
                    pr_auc,

                "ROC_AUC":
                    roc_auc,

                "Top10_TP":
                    top10_tp,

                "Top10_Precision":
                    top10_precision,

                "Top10_Recall":
                    top10_recall,

                "Top10_Lift":
                    top10_lift,

                "Train_Seconds":
                    seconds,
            })


fold_df = pd.DataFrame(
    fold_rows
)


# ============================================================
# 8. CV 汇总
# ============================================================

summary_df = (
    fold_df
    .groupby(
        [
            "Feature_Set",
            "Candidate"
        ],
        as_index=False
    )
    .agg(
        CV_Mean_PR_AUC=(
            "PR_AUC",
            "mean"
        ),

        CV_Std_PR_AUC=(
            "PR_AUC",
            "std"
        ),

        CV_Min_PR_AUC=(
            "PR_AUC",
            "min"
        ),

        CV_Mean_ROC_AUC=(
            "ROC_AUC",
            "mean"
        ),

        CV_Std_ROC_AUC=(
            "ROC_AUC",
            "std"
        ),

        CV_Mean_Top10_Recall=(
            "Top10_Recall",
            "mean"
        ),

        CV_Mean_Top10_Lift=(
            "Top10_Lift",
            "mean"
        ),

        CV_Total_Train_Seconds=(
            "Train_Seconds",
            "sum"
        ),
    )
)

# 稳健优先：
# 1) Mean PR-AUC
# 2) Min PR-AUC
# 3) Mean ROC-AUC
summary_df = (
    summary_df
    .sort_values(
        [
            "CV_Mean_PR_AUC",
            "CV_Min_PR_AUC",
            "CV_Mean_ROC_AUC"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)

print(
    "\n" + "=" * 90
)

print(
    "内部 CV 总表（PR-AUC 优先）"
)

print(
    "=" * 90
)

print(
    summary_df[
        [
            "Feature_Set",
            "Candidate",
            "CV_Mean_PR_AUC",
            "CV_Std_PR_AUC",
            "CV_Min_PR_AUC",
            "CV_Mean_ROC_AUC",
            "CV_Mean_Top10_Recall",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 9. 冻结内部 CV 最优 LightGBM
# ============================================================

best_cv = (
    summary_df.iloc[0]
)

best_feature_set = (
    best_cv[
        "Feature_Set"
    ]
)

best_candidate = (
    best_cv[
        "Candidate"
    ]
)

best_dropped = (
    FEATURE_SETS[
        best_feature_set
    ]
)

best_features = [
    c
    for c in all_features
    if c not in best_dropped
]

best_cat_features = [
    c
    for c in ALL_CATEGORICAL_FEATURES
    if c in best_features
]

best_cfg = next(
    cfg
    for cfg in LGB_CONFIGS
    if cfg["name"]
    == best_candidate
)

print(
    "\n[内部 CV 胜出]"
)

print(
    f"Feature Set = "
    f"{best_feature_set}"
)

print(
    f"Candidate = "
    f"{best_candidate}"
)

print(
    f"Mean PR-AUC = "
    f"{best_cv['CV_Mean_PR_AUC']:.6f}"
)

print(
    f"Mean ROC-AUC = "
    f"{best_cv['CV_Mean_ROC_AUC']:.6f}"
)


# ============================================================
# 10. 外层 Valid：只检查内部胜出候选
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "B. 外层 Validation 一次确认"
)

print(
    "=" * 90
)

X_train_final = (
    train[
        best_features
    ].copy()
)

X_valid_final = (
    valid[
        best_features
    ].copy()
)

Xtr_lgb, Xva_lgb = (
    make_lgb_frames(
        X_train_final,
        X_valid_final,
        best_cat_features
    )
)

final_lgb = create_model(
    best_cfg
)

start = time.time()

final_lgb.fit(
    Xtr_lgb,
    y_train_all,
    categorical_feature=
        best_cat_features
)

outer_seconds = (
    time.time()
    - start
)

outer_prob = (
    final_lgb
    .predict_proba(
        Xva_lgb
    )[:, 1]
)

outer_pr = (
    average_precision_score(
        y_valid_outer,
        outer_prob
    )
)

outer_roc = (
    roc_auc_score(
        y_valid_outer,
        outer_prob
    )
)

(
    outer_top10_tp,
    outer_top10_precision,
    outer_top10_recall,
    outer_top10_lift,
) = topk_metrics(
    y_valid_outer,
    outer_prob,
    0.10
)

# 外层 Valid 前半 / 后半稳定性
mid = (
    len(valid)
    // 2
)

early_pr = (
    average_precision_score(
        y_valid_outer.iloc[:mid],
        outer_prob[:mid]
    )
)

late_pr = (
    average_precision_score(
        y_valid_outer.iloc[mid:],
        outer_prob[mid:]
    )
)

early_roc = (
    roc_auc_score(
        y_valid_outer.iloc[:mid],
        outer_prob[:mid]
    )
)

late_roc = (
    roc_auc_score(
        y_valid_outer.iloc[mid:],
        outer_prob[mid:]
    )
)

outer_result = {
    "Feature_Set":
        best_feature_set,

    "Candidate":
        best_candidate,

    "Feature_Count":
        len(best_features),

    "Dropped_Features":
        ",".join(
            best_dropped
        ),

    "Outer_PR_AUC":
        outer_pr,

    "Outer_ROC_AUC":
        outer_roc,

    "Outer_Top10_TP":
        outer_top10_tp,

    "Outer_Top10_Precision":
        outer_top10_precision,

    "Outer_Top10_Recall":
        outer_top10_recall,

    "Outer_Top10_Lift":
        outer_top10_lift,

    "Early_PR_AUC":
        early_pr,

    "Late_PR_AUC":
        late_pr,

    "PR_Stability_Gap":
        abs(
            late_pr
            - early_pr
        ),

    "Early_ROC_AUC":
        early_roc,

    "Late_ROC_AUC":
        late_roc,

    "ROC_Stability_Gap":
        abs(
            late_roc
            - early_roc
        ),

    "Train_Seconds":
        outer_seconds,
}

outer_df = pd.DataFrame(
    [outer_result]
)

print(
    f"Outer PR-AUC  = "
    f"{outer_pr:.6f}"
)

print(
    f"Outer ROC-AUC = "
    f"{outer_roc:.6f}"
)

print(
    f"Early PR-AUC  = "
    f"{early_pr:.6f}"
)

print(
    f"Late PR-AUC   = "
    f"{late_pr:.6f}"
)

print(
    f"PR gap        = "
    f"{abs(late_pr-early_pr):.6f}"
)

print(
    f"Top10 Recall  = "
    f"{outer_top10_recall:.6f}"
)


# ============================================================
# 11. 特征重要性
# ============================================================

importance_df = pd.DataFrame({
    "Feature":
        best_features,

    "Importance":
        final_lgb
        .feature_importances_,
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 12. 保存
# ============================================================

fold_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_cv_fold_results.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_cv_summary.csv"
)

outer_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_outer_valid_result.csv"
)

pred_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_outer_valid_predictions.csv"
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_feature_importance.csv"
)

config_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_best_config.json"
)

text_summary_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_summary.txt"
)

fold_df.to_csv(
    fold_path,
    index=False,
    encoding="utf-8-sig"
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

outer_df.to_csv(
    outer_path,
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "y_true":
        y_valid_outer.values,

    "LightGBM_Probability":
        outer_prob,
}).to_csv(
    pred_path,
    index=False,
    encoding="utf-8-sig"
)

importance_df.to_csv(
    importance_path,
    index=False,
    encoding="utf-8-sig"
)

best_config_payload = {
    "random_state":
        RANDOM_STATE,

    "feature_set":
        best_feature_set,

    "dropped_features":
        best_dropped,

    "features":
        best_features,

    "categorical_features":
        best_cat_features,

    "config":
        best_cfg,

    "internal_cv": {
        "mean_pr_auc":
            float(
                best_cv[
                    "CV_Mean_PR_AUC"
                ]
            ),

        "std_pr_auc":
            float(
                best_cv[
                    "CV_Std_PR_AUC"
                ]
            ),

        "min_pr_auc":
            float(
                best_cv[
                    "CV_Min_PR_AUC"
                ]
            ),

        "mean_roc_auc":
            float(
                best_cv[
                    "CV_Mean_ROC_AUC"
                ]
            ),
    },

    "outer_validation": {
        "pr_auc":
            float(
                outer_pr
            ),

        "roc_auc":
            float(
                outer_roc
            ),

        "early_pr_auc":
            float(
                early_pr
            ),

        "late_pr_auc":
            float(
                late_pr
            ),

        "pr_stability_gap":
            float(
                abs(
                    late_pr
                    - early_pr
                )
            ),
    },
}

with open(
    config_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        best_config_payload,
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    text_summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "银行贷款审批预测 - "
        "LightGBM 优化 V1.0\n"
    )

    f.write(
        "=" * 78
        + "\n"
    )

    f.write(
        f"RANDOM_STATE = "
        f"{RANDOM_STATE}\n"
    )

    f.write(
        "\nSelection rule:\n"
        "1. Internal expanding-window CV Mean PR-AUC\n"
        "2. Internal Min PR-AUC\n"
        "3. Mean ROC-AUC\n"
    )

    f.write(
        "\nInternal CV winner:\n"
    )

    f.write(
        f"Feature_Set = "
        f"{best_feature_set}\n"
    )

    f.write(
        f"Candidate = "
        f"{best_candidate}\n"
    )

    f.write(
        f"CV Mean PR-AUC = "
        f"{best_cv['CV_Mean_PR_AUC']:.8f}\n"
    )

    f.write(
        f"CV Std PR-AUC = "
        f"{best_cv['CV_Std_PR_AUC']:.8f}\n"
    )

    f.write(
        f"CV Min PR-AUC = "
        f"{best_cv['CV_Min_PR_AUC']:.8f}\n"
    )

    f.write(
        f"CV Mean ROC-AUC = "
        f"{best_cv['CV_Mean_ROC_AUC']:.8f}\n"
    )

    f.write(
        "\nOuter validation (one check):\n"
    )

    f.write(
        f"PR-AUC = "
        f"{outer_pr:.8f}\n"
    )

    f.write(
        f"ROC-AUC = "
        f"{outer_roc:.8f}\n"
    )

    f.write(
        f"Early PR-AUC = "
        f"{early_pr:.8f}\n"
    )

    f.write(
        f"Late PR-AUC = "
        f"{late_pr:.8f}\n"
    )

    f.write(
        f"PR Stability Gap = "
        f"{abs(late_pr-late_pr + late_pr-early_pr):.8f}\n"
    )

    f.write(
        f"Top10 Recall = "
        f"{outer_top10_recall:.8f}\n"
    )

    f.write(
        "\nReference current champion:\n"
        "XGBoost X08 + DROP_CITY\n"
        "Validation PR-AUC = 0.111975\n"
        "Validation ROC-AUC = 0.844654\n"
    )

    f.write(
        "\nDecision principle:\n"
        "- LightGBM does not need to beat XGBoost alone to enter Stacking.\n"
        "- It only needs reasonable validation quality and complementary predictions.\n"
        "- The next step compares XGB+CAT Stacking versus XGB+CAT+LGB Stacking.\n"
    )

print(
    "\n✅ 已保存："
)

print(
    "1. lightgbm_cv_fold_results.csv"
)

print(
    "2. lightgbm_cv_summary.csv"
)

print(
    "3. lightgbm_outer_valid_result.csv"
)

print(
    "4. lightgbm_outer_valid_predictions.csv"
)

print(
    "5. lightgbm_feature_importance.csv"
)

print(
    "6. lightgbm_best_config.json"
)

print(
    "7. lightgbm_summary.txt"
)

print(
    "\n" + "=" * 90
)

print(
    "LightGBM V1.0 完成"
)

print(
    "=" * 90
)
