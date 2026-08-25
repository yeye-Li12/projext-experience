# -*- coding: utf-8 -*-
"""
银行贷款审批预测：V1.1 Final Refit + Submission
==============================================

最终方案已经冻结：
------------------------------------------------
模型：
XGBoost XGB_X08 + DROP_CITY

主评价指标：
PR-AUC

参考指标：
ROC-AUC

最终模型参数：
n_estimators=800
max_depth=3
learning_rate=0.025
min_child_weight=10
subsample=0.9
colsample_bytree=0.9
gamma=0.1
reg_lambda=10.0
reg_alpha=0.0
scale_pos_weight=20
random_state=42

最终删除特征：
- City_Code
- City_Category

------------------------------------------------
FINAL REFIT 原则：
1. 从原始 train.csv 开始；
2. 不再进行 80/20 Train / Validation 划分；
3. 使用全部 43,980 条有标签数据 fit V1.4 预处理器；
4. 使用全部 43,980 条数据训练冻结后的 XGBoost；
5. 原始 Test 只能使用该预处理器 transform；
6. Test 不参与任何：
   - 中位数计算
   - P99 计算
   - 类别编码 fit
   - Loan 填充值统计
   - 模型训练
   - 参数选择
7. 输出 probability 分数供 PR-AUC / ROC-AUC 排名评估。

重要：
scale_pos_weight=20 后的 predict_proba 是模型分数意义上的概率输出，
不应直接解释为“真实业务获批概率”。
但 PR-AUC / ROC-AUC 都只依赖排序，因此直接提交 predict_proba 是合适的。
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "缺少 XGBoost，请先执行：\n"
        "pip install xgboost pandas numpy scikit-learn\n"
        f"\n原始错误：{e}"
    )


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_DIR = r"D:\深圳点宽\银行贷款审批"

RAW_TRAIN_PATH = os.path.join(
    PROJECT_DIR,
    "train.csv"
)

# 默认使用 test_features.csv
# 如果你原始测试集文件名不同，只修改这里
RAW_TEST_PATH = os.path.join(
    PROJECT_DIR,
    "test_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "output_v14",
    "final_v11"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42

TARGET = "Approved"
ID_COL = "ID"

EXPECTED_TRAIN_ROWS = 43980
EXPECTED_TEST_ROWS = 25733

DROP_CITY = [
    "City_Code",
    "City_Category"
]

# V1.4 中明确排除
EXCLUDED_MODEL_COLS = [
    "Employer_Code"
]

FINAL_CATEGORICAL_FEATURES = [
    "Gender",
    "Employer_Category1",
    "Employer_Category2",
    "Customer_Existing_Primary_Bank_Code",
    "Primary_Bank_Type",
    "Source",
    "Source_Category",
    "Age_Bin",
]


# ============================================================
# 2. 文件读取
# ============================================================

def load_csv_auto(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n找不到文件：\n{path}\n"
        )

    encodings = [
        "gbk",
        "gb18030",
        "utf-8-sig",
        "utf-8",
    ]

    last_error = None

    for enc in encodings:

        try:
            df = pd.read_csv(
                path,
                encoding=enc
            )

            print(
                f"✅ 读取成功：{path}\n"
                f"   encoding={enc}, shape={df.shape}"
            )

            return df

        except Exception as e:
            last_error = e

    raise last_error


# ============================================================
# 3. V1.4 Final FeatureEngineer
# ============================================================

class FinalFeatureEngineer:

    def __init__(self):

        self.fitted = False

        self.num_medians = {}
        self.num_upper_limits = {}

        self.age_median = np.nan

        self.cat_cols = []
        self.cat_encoders = {}

        self.loan_amount_by_period = {}

        self.loan_amount_global_median = np.nan
        self.loan_period_global_median = np.nan

        self.feature_cols = []


    # --------------------------------------------------------
    # 3.1 基础清洗
    # --------------------------------------------------------

    @staticmethod
    def basic_clean(df):

        df = df.copy()

        # ----------------------------
        # DOB -> datetime
        # ----------------------------

        if "DOB" in df.columns:

            df["DOB"] = pd.to_datetime(
                df["DOB"],
                format="%d/%m/%y",
                errors="coerce"
            )

            # 两位数年份修正
            df["DOB"] = df["DOB"].apply(
                lambda dt:
                dt - pd.DateOffset(years=100)
                if (
                    pd.notnull(dt)
                    and dt.year > 2020
                )
                else dt
            )


        # ----------------------------
        # Lead_Creation_Date
        # ----------------------------

        if "Lead_Creation_Date" in df.columns:

            df[
                "Lead_Creation_Date"
            ] = pd.to_datetime(
                df[
                    "Lead_Creation_Date"
                ],
                dayfirst=True,
                errors="coerce"
            )


        # ----------------------------
        # Age
        # ----------------------------

        if (
            "DOB" in df.columns
            and
            "Lead_Creation_Date" in df.columns
        ):

            lead = df[
                "Lead_Creation_Date"
            ]

            dob = df[
                "DOB"
            ]

            not_yet_birthday = (
                (
                    lead.dt.month
                    <
                    dob.dt.month
                )
                |
                (
                    (
                        lead.dt.month
                        ==
                        dob.dt.month
                    )
                    &
                    (
                        lead.dt.day
                        <
                        dob.dt.day
                    )
                )
            )

            df["Age"] = (
                lead.dt.year
                -
                dob.dt.year
                -
                not_yet_birthday.astype(int)
            )

            df.loc[
                (
                    df["Age"].isna()
                )
                |
                (
                    df["Age"] < 18
                )
                |
                (
                    df["Age"] > 65
                ),
                "Age"
            ] = np.nan


        # ----------------------------
        # 删除 T1 / 不可上线字段
        # ----------------------------

        drop_cols = [
            "Contacted",
            "Interest_Rate",
            "EMI",
            "DOB",
        ]

        df.drop(
            columns=[
                c
                for c in drop_cols
                if c in df.columns
            ],
            inplace=True,
            errors="ignore"
        )


        # Employer_Code 在 V1.4 被主动排除
        df.drop(
            columns=[
                c
                for c in EXCLUDED_MODEL_COLS
                if c in df.columns
            ],
            inplace=True,
            errors="ignore"
        )


        # Employer_Category2 特殊缺失
        if (
            "Employer_Category2"
            in df.columns
        ):

            df[
                "Employer_Category2"
            ] = (
                df[
                    "Employer_Category2"
                ]
                .fillna("UNKNOWN")
                .astype(str)
            )

        return df


    # --------------------------------------------------------
    # 3.2 Fit：只能在完整 train 上执行
    # --------------------------------------------------------

    def fit(self, train_df):

        df = train_df.copy()

        # 数值缺失统计量
        self.num_medians = {
            "Monthly_Income":
                df[
                    "Monthly_Income"
                ].median(),

            "Existing_EMI":
                df[
                    "Existing_EMI"
                ].median(),

            "Var1":
                df[
                    "Var1"
                ].median(),
        }


        self.age_median = (
            df[
                "Age"
            ].median()
        )


        # P99 缩尾
        self.num_upper_limits = {
            "Monthly_Income":
                df[
                    "Monthly_Income"
                ].quantile(0.99),

            "Existing_EMI":
                df[
                    "Existing_EMI"
                ].quantile(0.99),
        }


        # 类别字段
        self.cat_cols = [
            "Gender",
            "City_Category",
            "Employer_Category1",
            "Employer_Category2",
            "Customer_Existing_Primary_Bank_Code",
            "Primary_Bank_Type",
            "Source",
            "Source_Category",
            "City_Code",
        ]

        self.cat_cols = [
            c
            for c in self.cat_cols
            if c in df.columns
        ]


        # 类别编码器只在 Train fit
        self.cat_encoders = {}

        for col in self.cat_cols:

            values = (
                df[col]
                .fillna("UNKNOWN")
                .astype(str)
            )

            le = LabelEncoder()

            le.fit(
                values.unique()
            )

            self.cat_encoders[
                col
            ] = le


        # Loan 条件中位数
        self.loan_amount_by_period = (
            df.groupby(
                "Loan_Period"
            )[
                "Loan_Amount"
            ]
            .median()
            .to_dict()
        )


        self.loan_amount_global_median = (
            df[
                "Loan_Amount"
            ]
            .median()
        )


        self.loan_period_global_median = (
            df[
                "Loan_Period"
            ]
            .median()
        )


        self.fitted = True

        return self


    # --------------------------------------------------------
    # 3.3 Transform
    # --------------------------------------------------------

    def transform(
        self,
        df,
        keep_id=False,
        keep_target=False
    ):

        if not self.fitted:

            raise ValueError(
                "FeatureEngineer 尚未 fit"
            )

        df = df.copy()

        ids = None
        target = None


        if (
            keep_id
            and
            ID_COL in df.columns
        ):
            ids = df[
                ID_COL
            ].copy()


        if (
            keep_target
            and
            TARGET in df.columns
        ):
            target = (
                df[
                    TARGET
                ]
                .astype(int)
                .copy()
            )


        # ----------------------------
        # 数值缺失
        # ----------------------------

        for col in [
            "Monthly_Income",
            "Existing_EMI",
            "Var1",
        ]:

            df[col] = (
                df[col]
                .fillna(
                    self.num_medians[col]
                )
            )


        # ----------------------------
        # Age
        # ----------------------------

        age_missing_mask = (
            df[
                "Age"
            ].isna()
        )


        df[
            "Age_Missing"
        ] = (
            age_missing_mask
            .astype(int)
        )


        df[
            "Age"
        ] = (
            df[
                "Age"
            ]
            .fillna(
                self.age_median
            )
        )


        # ----------------------------
        # Loan 缺失
        # ----------------------------

        loan_amount_missing = (
            df[
                "Loan_Amount"
            ].isna()
        )

        loan_period_missing = (
            df[
                "Loan_Period"
            ].isna()
        )


        df[
            "Loan_Info_Missing"
        ] = (
            loan_amount_missing
            |
            loan_period_missing
        ).astype(int)


        # Loan_Amount：
        # 优先按 Loan_Period 训练集条件中位数填充
        amt_miss = (
            df[
                "Loan_Amount"
            ].isna()
        )

        period_exists = (
            df[
                "Loan_Period"
            ].notna()
        )


        mapped_amount = (
            df[
                "Loan_Period"
            ]
            .map(
                self.loan_amount_by_period
            )
        )


        df.loc[
            amt_miss
            &
            period_exists,
            "Loan_Amount"
        ] = (
            mapped_amount[
                amt_miss
                &
                period_exists
            ]
        )


        df[
            "Loan_Amount"
        ] = (
            df[
                "Loan_Amount"
            ]
            .fillna(
                self.loan_amount_global_median
            )
        )


        df[
            "Loan_Period"
        ] = (
            df[
                "Loan_Period"
            ]
            .fillna(
                self.loan_period_global_median
            )
        )


        # ----------------------------
        # 类别缺失
        # ----------------------------

        for col in self.cat_cols:

            if col in df.columns:

                df[col] = (
                    df[col]
                    .fillna("UNKNOWN")
                    .astype(str)
                )


        # ----------------------------
        # P99
        # ----------------------------

        df[
            "Monthly_Income"
        ] = (
            df[
                "Monthly_Income"
            ]
            .clip(
                upper=
                self.num_upper_limits[
                    "Monthly_Income"
                ]
            )
        )


        df[
            "Existing_EMI"
        ] = (
            df[
                "Existing_EMI"
            ]
            .clip(
                upper=
                self.num_upper_limits[
                    "Existing_EMI"
                ]
            )
        )


        # ----------------------------
        # 衍生特征
        # ----------------------------

        df[
            "EMI_Zero_Flag"
        ] = (
            df[
                "Existing_EMI"
            ]
            ==
            0
        ).astype(int)


        df[
            "DTI"
        ] = (
            df[
                "Existing_EMI"
            ]
            /
            (
                df[
                    "Monthly_Income"
                ]
                +
                1e-5
            )
        ).clip(
            0,
            5
        )


        df[
            "Log_Income"
        ] = (
            np.log1p(
                df[
                    "Monthly_Income"
                ]
            )
        )


        df[
            "Loan_to_Income"
        ] = (
            df[
                "Loan_Amount"
            ]
            /
            (
                df[
                    "Monthly_Income"
                ]
                +
                1e-5
            )
        ).clip(
            0,
            100
        )


        # ----------------------------
        # Age Bin
        # ----------------------------

        df[
            "Age_Bin"
        ] = pd.cut(
            df[
                "Age"
            ],
            bins=[
                18,
                25,
                35,
                45,
                55,
                66
            ],
            right=False,
            labels=[
                0,
                1,
                2,
                3,
                4
            ]
        ).astype(float)


        df.loc[
            age_missing_mask,
            "Age_Bin"
        ] = -1


        df[
            "Age_Bin"
        ] = (
            df[
                "Age_Bin"
            ]
            .astype(int)
        )


        # ----------------------------
        # Label Encoding
        # ----------------------------

        for col in self.cat_cols:

            if col not in df.columns:
                continue

            le = (
                self.cat_encoders[col]
            )

            mapping = {
                cls: i
                for i, cls
                in enumerate(
                    le.classes_
                )
            }

            df[col] = (
                df[col]
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )


        # ----------------------------
        # 只保留 V1.4 建模特征
        # ----------------------------

        feature_cols = [
            "Monthly_Income",
            "Existing_EMI",
            "Var1",
            "Loan_Amount",
            "Loan_Period",
            "Loan_Info_Missing",
            "Age",
            "Age_Missing",
            "DTI",
            "Log_Income",
            "Loan_to_Income",
            "EMI_Zero_Flag",
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


        feature_cols = [
            c
            for c in feature_cols
            if c in df.columns
        ]


        self.feature_cols = list(
            feature_cols
        )


        output = df[
            feature_cols
        ].copy()


        if (
            keep_target
            and target is not None
        ):

            output[
                TARGET
            ] = target.values


        if (
            keep_id
            and ids is not None
        ):

            output.insert(
                0,
                ID_COL,
                ids.values
            )


        return output


    # --------------------------------------------------------
    # 3.4 导出 fit stats
    # --------------------------------------------------------

    def export_stats(self):

        return {
            "num_medians":
                {
                    k:
                        (
                            None
                            if pd.isna(v)
                            else float(v)
                        )
                    for k, v
                    in self.num_medians.items()
                },

            "age_median":
                (
                    None
                    if pd.isna(
                        self.age_median
                    )
                    else float(
                        self.age_median
                    )
                ),

            "num_upper_limits":
                {
                    k:
                        (
                            None
                            if pd.isna(v)
                            else float(v)
                        )
                    for k, v
                    in self.num_upper_limits.items()
                },

            "loan_amount_global_median":
                (
                    None
                    if pd.isna(
                        self.loan_amount_global_median
                    )
                    else float(
                        self.loan_amount_global_median
                    )
                ),

            "loan_period_global_median":
                (
                    None
                    if pd.isna(
                        self.loan_period_global_median
                    )
                    else float(
                        self.loan_period_global_median
                    )
                ),

            "categorical_classes":
                {
                    col:
                        [
                            str(x)
                            for x in
                            self.cat_encoders[
                                col
                            ].classes_
                        ]
                    for col
                    in self.cat_encoders
                },
        }


# ============================================================
# 4. 读取原始数据
# ============================================================

print(
    "=" * 92
)

print(
    "V1.1 Final Refit + Submission"
)

print(
    "=" * 92
)


raw_train = load_csv_auto(
    RAW_TRAIN_PATH
)

raw_test = load_csv_auto(
    RAW_TEST_PATH
)


# ============================================================
# 5. 基础检查
# ============================================================

print(
    "\n[原始数据检查]"
)


if TARGET not in raw_train.columns:

    raise ValueError(
        "原始 Train 缺少 Approved 标签"
    )


if TARGET in raw_test.columns:

    raise ValueError(
        "原始 Test 不应该包含 Approved"
    )


if ID_COL not in raw_train.columns:

    print(
        "⚠️ Train 中没有 ID，"
        "不影响最终训练。"
    )


if ID_COL not in raw_test.columns:

    raise ValueError(
        "原始 Test 缺少 ID，"
        "无法生成 submission"
    )


if len(raw_train) != EXPECTED_TRAIN_ROWS:

    print(
        f"⚠️ Train 行数={len(raw_train)}，"
        f"预期={EXPECTED_TRAIN_ROWS}"
    )


if len(raw_test) != EXPECTED_TEST_ROWS:

    print(
        f"⚠️ Test 行数={len(raw_test)}，"
        f"预期={EXPECTED_TEST_ROWS}"
    )


if raw_test[
    ID_COL
].duplicated().any():

    raise ValueError(
        "Test ID 存在重复"
    )


if raw_test[
    ID_COL
].isna().any():

    raise ValueError(
        "Test ID 存在缺失"
    )


print(
    f"Train rows = {len(raw_train)}"
)

print(
    f"Train positives = "
    f"{int(raw_train[TARGET].sum())}"
)

print(
    f"Train positive rate = "
    f"{raw_train[TARGET].mean():.8f}"
)

print(
    f"Test rows = {len(raw_test)}"
)


# ============================================================
# 6. 基础清洗
# ============================================================

train_clean = (
    FinalFeatureEngineer
    .basic_clean(
        raw_train
    )
)

test_clean = (
    FinalFeatureEngineer
    .basic_clean(
        raw_test
    )
)


# ============================================================
# 7. 完整 43,980 条 Fit 预处理器
# ============================================================

print(
    "\n" + "=" * 92
)

print(
    "A. Fit V1.4 Final Preprocessor on FULL Train"
)

print(
    "=" * 92
)


fe = FinalFeatureEngineer()

fe.fit(
    train_clean
)


train_processed = (
    fe.transform(
        train_clean,
        keep_target=True
    )
)


test_processed = (
    fe.transform(
        test_clean,
        keep_id=True
    )
)


print(
    f"Final processed train: "
    f"{train_processed.shape}"
)

print(
    f"Final processed test : "
    f"{test_processed.shape}"
)


# ============================================================
# 8. 数据质量检查
# ============================================================

feature_cols_full = [
    c
    for c
    in train_processed.columns
    if c != TARGET
]


test_feature_cols_full = [
    c
    for c
    in test_processed.columns
    if c != ID_COL
]


if (
    feature_cols_full
    !=
    test_feature_cols_full
):

    raise ValueError(
        "\nTrain / Test 处理后特征顺序不一致。\n"
        f"Train={feature_cols_full}\n"
        f"Test={test_feature_cols_full}"
    )


train_nan = (
    train_processed[
        feature_cols_full
    ]
    .isna()
    .sum()
    .sum()
)

test_nan = (
    test_processed[
        test_feature_cols_full
    ]
    .isna()
    .sum()
    .sum()
)


if train_nan > 0:

    raise ValueError(
        f"Final Train 仍存在 "
        f"{train_nan} 个 NaN"
    )


if test_nan > 0:

    raise ValueError(
        f"Final Test 仍存在 "
        f"{test_nan} 个 NaN"
    )


train_inf = int(
    np.isinf(
        train_processed[
            feature_cols_full
        ]
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .to_numpy()
    ).sum()
)


test_inf = int(
    np.isinf(
        test_processed[
            test_feature_cols_full
        ]
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .to_numpy()
    ).sum()
)


if train_inf > 0:

    raise ValueError(
        f"Final Train 存在 "
        f"{train_inf} 个 Inf"
    )


if test_inf > 0:

    raise ValueError(
        f"Final Test 存在 "
        f"{test_inf} 个 Inf"
    )


print(
    "✅ NaN = 0"
)

print(
    "✅ Inf = 0"
)

print(
    "✅ Train/Test 特征顺序一致"
)


# ============================================================
# 9. 删除 City 特征
# ============================================================

final_features = [
    c
    for c
    in feature_cols_full
    if c not in DROP_CITY
]


X_train = (
    train_processed[
        final_features
    ].copy()
)


y_train = (
    train_processed[
        TARGET
    ]
    .astype(int)
    .copy()
)


X_test = (
    test_processed[
        final_features
    ].copy()
)


test_ids = (
    test_processed[
        ID_COL
    ].copy()
)


print(
    "\n[最终模型特征]"
)

print(
    f"Feature count = "
    f"{len(final_features)}"
)

for i, col in enumerate(
    final_features,
    start=1
):

    print(
        f"{i:02d}. {col}"
    )


if len(final_features) != 20:

    print(
        f"⚠️ 当前最终特征数={len(final_features)}，"
        "此前预期为20，请检查原始字段。"
    )


# ============================================================
# 10. 最终 XGBoost 全量训练
# ============================================================

print(
    "\n" + "=" * 92
)

print(
    "B. Final Refit: XGBoost X08 + DROP_CITY"
)

print(
    "=" * 92
)


final_model = XGBClassifier(
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


final_model.fit(
    X_train,
    y_train
)


print(
    "✅ Final XGBoost training complete"
)


# ============================================================
# 11. Test prediction
# ============================================================

print(
    "\n" + "=" * 92
)

print(
    "C. Predict Test"
)

print(
    "=" * 92
)


test_probability = (
    final_model
    .predict_proba(
        X_test
    )[:, 1]
)


# 数值安全
test_probability = (
    np.clip(
        test_probability,
        1e-12,
        1 - 1e-12
    )
)


# percentile rank 仅做诊断
test_rank = (
    pd.Series(
        test_probability
    )
    .rank(
        method="average",
        pct=True
    )
    .to_numpy()
)


print(
    f"Probability min  = "
    f"{test_probability.min():.10f}"
)

print(
    f"Probability mean = "
    f"{test_probability.mean():.10f}"
)

print(
    f"Probability max  = "
    f"{test_probability.max():.10f}"
)


# ============================================================
# 12. Submission
# ============================================================

submission = pd.DataFrame({
    "ID":
        test_ids.values,

    "probability":
        test_probability,
})


if len(submission) != len(
    raw_test
):

    raise ValueError(
        "Submission 行数和 Test 不一致"
    )


if submission[
    "ID"
].duplicated().any():

    raise ValueError(
        "Submission ID 重复"
    )


if submission[
    "probability"
].isna().any():

    raise ValueError(
        "Submission probability 存在 NaN"
    )


if not submission[
    "probability"
].between(
    0,
    1
).all():

    raise ValueError(
        "Submission probability 不在 [0,1]"
    )


submission_path = os.path.join(
    OUTPUT_DIR,
    "submission_xgb_x08_drop_city_v11.csv"
)


submission.to_csv(
    submission_path,
    index=False,
    encoding="utf-8"
)


print(
    f"\n✅ 正式 submission：\n"
    f"{submission_path}"
)


# ============================================================
# 13. 保存处理后数据
# ============================================================

processed_train_path = os.path.join(
    OUTPUT_DIR,
    "full_train_processed_v11.csv"
)


processed_test_path = os.path.join(
    OUTPUT_DIR,
    "test_processed_v11.csv"
)


train_processed.to_csv(
    processed_train_path,
    index=False,
    encoding="utf-8-sig"
)


test_processed.to_csv(
    processed_test_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 保存 Test 诊断分数
# ============================================================

test_score_df = pd.DataFrame({
    "ID":
        test_ids.values,

    "XGBoost_Probability":
        test_probability,

    "Percentile_Rank":
        test_rank,
})


test_score_path = os.path.join(
    OUTPUT_DIR,
    "test_prediction_diagnostics_v11.csv"
)


test_score_df.to_csv(
    test_score_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. 特征重要性
# ============================================================

importance_df = pd.DataFrame({
    "Feature":
        final_features,

    "Importance":
        final_model.feature_importances_,
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


importance_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_final_feature_importance_v11.csv"
)


importance_df.to_csv(
    importance_path,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n[Top 15 Feature Importance]"
)

print(
    importance_df
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 16. 保存预处理统计量
# ============================================================

fit_stats = (
    fe.export_stats()
)


fit_stats_path = os.path.join(
    OUTPUT_DIR,
    "final_preprocess_fit_stats_v11.json"
)


with open(
    fit_stats_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        fit_stats,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 17. 保存模型配置
# ============================================================

config = {
    "version":
        "V1.1 Final Refit",

    "random_state":
        RANDOM_STATE,

    "selection_metric":
        "PR-AUC",

    "reference_metric":
        "ROC-AUC",

    "training_data":
        {
            "rows":
                int(
                    len(
                        train_processed
                    )
                ),

            "positives":
                int(
                    y_train.sum()
                ),

            "positive_rate":
                float(
                    y_train.mean()
                ),
        },

    "preprocessing":
        {
            "fit_on":
                "FULL labeled train only",

            "test_role":
                "transform only",

            "version":
                "V1.4 logic, FINAL mode without train/valid split",
        },

    "model":
        {
            "type":
                "XGBoost",

            "candidate":
                "XGB_X08",

            "drop_features":
                DROP_CITY,

            "feature_count":
                len(
                    final_features
                ),

            "features":
                final_features,

            "params":
                {
                    "n_estimators":
                        800,

                    "max_depth":
                        3,

                    "learning_rate":
                        0.025,

                    "min_child_weight":
                        10,

                    "subsample":
                        0.9,

                    "colsample_bytree":
                        0.9,

                    "gamma":
                        0.1,

                    "reg_lambda":
                        10.0,

                    "reg_alpha":
                        0.0,

                    "scale_pos_weight":
                        20,

                    "random_state":
                        RANDOM_STATE,
                },
        },

    "development_validation_reference":
        {
            "PR_AUC":
                0.111975,

            "ROC_AUC":
                0.844654,

            "note":
                (
                    "这些指标来自模型开发阶段的独立时间验证集，"
                    "不是 Final Refit 后重新计算的训练指标。"
                ),
        },

    "submission":
        {
            "columns":
                [
                    "ID",
                    "probability"
                ],

            "rows":
                int(
                    len(
                        submission
                    )
                ),
        },
}


config_path = os.path.join(
    OUTPUT_DIR,
    "final_model_config_v11.json"
)


with open(
    config_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 18. 保存最终摘要
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "final_refit_summary_v11.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "银行贷款审批预测 - "
        "V1.1 Final Refit + Submission\n"
    )

    f.write(
        "=" * 80
        +
        "\n"
    )

    f.write(
        f"RANDOM_STATE = "
        f"{RANDOM_STATE}\n"
    )

    f.write(
        "\nFinal model:\n"
    )

    f.write(
        "XGBoost XGB_X08 + DROP_CITY\n"
    )

    f.write(
        "Dropped features:\n"
        "- City_Code\n"
        "- City_Category\n"
    )

    f.write(
        f"\nFull labeled training rows = "
        f"{len(train_processed)}\n"
    )

    f.write(
        f"Positive samples = "
        f"{int(y_train.sum())}\n"
    )

    f.write(
        f"Positive rate = "
        f"{y_train.mean():.8f}\n"
    )

    f.write(
        f"Final feature count = "
        f"{len(final_features)}\n"
    )

    f.write(
        "\nDevelopment-stage validation reference:\n"
    )

    f.write(
        "PR-AUC = 0.111975\n"
    )

    f.write(
        "ROC-AUC = 0.844654\n"
    )

    f.write(
        "\nImportant:\n"
        "The above validation scores were obtained BEFORE Final Refit "
        "on the held-out temporal validation set.\n"
        "After the model was frozen, all 43,980 labeled rows were used "
        "for Final Refit and no new validation score was calculated.\n"
    )

    f.write(
        "\nFinal preprocessing:\n"
        "- V1.4 feature engineering logic\n"
        "- No 80/20 split\n"
        "- Preprocessing fit on full labeled Train only\n"
        "- Test transform only\n"
        "- Test never participates in fit statistics or training\n"
    )

    f.write(
        "\nFinal XGBoost params:\n"
        "n_estimators=800\n"
        "max_depth=3\n"
        "learning_rate=0.025\n"
        "min_child_weight=10\n"
        "subsample=0.9\n"
        "colsample_bytree=0.9\n"
        "gamma=0.1\n"
        "reg_lambda=10.0\n"
        "reg_alpha=0.0\n"
        "scale_pos_weight=20\n"
        "random_state=42\n"
    )

    f.write(
        "\nTest prediction:\n"
    )

    f.write(
        f"Rows = "
        f"{len(submission)}\n"
    )

    f.write(
        f"Probability min = "
        f"{test_probability.min():.10f}\n"
    )

    f.write(
        f"Probability mean = "
        f"{test_probability.mean():.10f}\n"
    )

    f.write(
        f"Probability max = "
        f"{test_probability.max():.10f}\n"
    )

    f.write(
        "\nOfficial submission:\n"
    )

    f.write(
        f"{submission_path}\n"
    )

    f.write(
        "\nInterpretation note:\n"
        "Because scale_pos_weight=20 is used, model predict_proba should "
        "be treated primarily as a ranking score rather than a calibrated "
        "real-world approval probability. This is appropriate for PR-AUC "
        "and ROC-AUC evaluation because both metrics depend on ranking.\n"
    )


# ============================================================
# 19. 保存 Feature List
# ============================================================

feature_list_path = os.path.join(
    OUTPUT_DIR,
    "final_feature_list_v11.txt"
)


with open(
    feature_list_path,
    "w",
    encoding="utf-8"
) as f:

    for col in final_features:
        f.write(
            col
            +
            "\n"
        )


# ============================================================
# 20. 最终输出
# ============================================================

print(
    "\n" + "=" * 92
)

print(
    "V1.1 FINAL REFIT 完成"
)

print(
    "=" * 92
)

print(
    "\n生成文件："
)

print(
    "1. submission_xgb_x08_drop_city_v11.csv"
)

print(
    "2. final_refit_summary_v11.txt"
)

print(
    "3. test_prediction_diagnostics_v11.csv"
)

print(
    "4. xgboost_final_feature_importance_v11.csv"
)

print(
    "5. final_preprocess_fit_stats_v11.json"
)

print(
    "6. final_model_config_v11.json"
)

print(
    "7. final_feature_list_v11.txt"
)

print(
    "8. full_train_processed_v11.csv"
)

print(
    "9. test_processed_v11.csv"
)

print(
    "\n正式提交文件："
)

print(
    submission_path
)

print(
    "\n请将以下文件发给我做最终验收："
)

print(
    "1. submission_xgb_x08_drop_city_v11.csv"
)

print(
    "2. final_refit_summary_v11.txt"
)

print(
    "3. test_prediction_diagnostics_v11.csv"
)

print(
    "4. xgboost_final_feature_importance_v11.csv"
)

print(
    "5. final_model_config_v11.json"
)
