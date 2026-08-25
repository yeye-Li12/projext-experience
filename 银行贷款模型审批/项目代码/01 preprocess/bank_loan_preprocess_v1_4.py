# -*- coding: utf-8 -*-
"""
银行贷款审批预测：数据预处理与特征工程脚本 V1.4
================================================
V1.4 相对 V1.3 的主要修订：
1. 默认采用时间顺序 80% Train / 20% Validation，避免验证集过大。
2. Employer_Code 暂不入主模型：
   V1.3 结果中其验证集/测试集未见类别比例过高，时间泛化较差。
3. 合并 Loan_Amount / Loan_Period 的重复缺失标记为 Loan_Info_Missing。
4. Age 增加 Age_Missing，并使用“训练部分 Age 中位数”填补数值 Age；
   Age_Bin 对原本缺失年龄仍保留 -1 表示未知。
5. 不再生成在当前数据上恒为 0 的
   Monthly_Income_Missing / Existing_EMI_Missing / Var1_Missing。
6. 继续严格遵守：先做时间划分，再仅用训练部分拟合所有统计量与编码器。
7. 输出文件统一增加 _v14 后缀，避免覆盖 V1.3 结果。
8. 新增 split_summary_v14.csv 与 run_summary_v14.txt，方便结果验收。
说明：
- 当前脚本用于“模型选择/验证阶段”。
- 最终确定模型和参数之后，应再用完整 train.csv 重新 fit 预处理器和最终模型，
  然后重新 transform test_features.csv 并生成最终 submission.csv。
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
# ============================================================
# 配置区【已按您的本地路径完成修改】
# ============================================================
TRAIN_PATH = r"D:\深圳点宽\银行贷款审批\train.csv"
TEST_PATH = r"D:\深圳点宽\银行贷款审批\test_features.csv"
OUTPUT_DIR = r"D:\深圳点宽\银行贷款审批\output_v14"
# 主验证方案：严格按时间顺序，前 80% 训练、后 20% 验证
VALID_RATIO = 0.20
MIN_VALID_POSITIVES = 20
OUT_ENCODING = "utf-8-sig"
# 当前暂不进入主模型的高基数/时间泛化不稳定字段
EXCLUDED_MODEL_COLS = ["Employer_Code"]
# ============================================================
# T0 字段可得性 / 建模审计表
# ============================================================
FIELD_AVAILABILITY = [
    {"Feature": "ID", "是否入模": "❌", "预测时点": "T0", "原因": "主键，无直接预测意义"},
    {"Feature": "Gender", "是否入模": "✅", "预测时点": "T0", "原因": "申请时可得"},
    {"Feature": "DOB", "是否入模": "❌→衍生Age", "预测时点": "T0", "原因": "原始日期不直接入模，衍生 Age / Age_Bin"},
    {"Feature": "Lead_Creation_Date", "是否入模": "❌→仅排序/切分", "预测时点": "T0", "原因": "申请时间用于时间排序、切分与年龄计算"},
    {"Feature": "City_Code", "是否入模": "✅", "预测时点": "T0", "原因": "申请时可得"},
    {"Feature": "City_Category", "是否入模": "✅", "预测时点": "T0", "原因": "申请时可得"},
    {"Feature": "Employer_Code", "是否入模": "❌（V1.4主模型暂移除）", "预测时点": "T0",
     "原因": "V1.3 中验证/测试未见类别比例过高，时间泛化不稳定；后续可单独研究更稳健编码"},
    {"Feature": "Employer_Category1", "是否入模": "✅", "预测时点": "T0", "原因": "申请时可得"},
    {"Feature": "Employer_Category2", "是否入模": "✅", "预测时点": "T0", "原因": "申请时可得"},
    {"Feature": "Monthly_Income", "是否入模": "✅", "预测时点": "T0", "原因": "申请时可得"},
    {"Feature": "Customer_Existing_Primary_Bank_Code", "是否入模": "✅", "预测时点": "T0", "原因": "申请时可得"},
    {"Feature": "Primary_Bank_Type", "是否入模": "✅", "预测时点": "T0", "原因": "申请时可得"},
    {"Feature": "Source", "是否入模": "✅", "预测时点": "T0", "原因": "申请渠道，申请时可得"},
    {"Feature": "Source_Category", "是否入模": "✅", "预测时点": "T0", "原因": "申请渠道分类，申请时可得"},
    {"Feature": "Existing_EMI", "是否入模": "✅", "预测时点": "T0", "原因": "现有负债，申请时可得"},
    {"Feature": "Loan_Amount", "是否入模": "✅", "预测时点": "T0", "原因": "申请金额"},
    {"Feature": "Loan_Period", "是否入模": "✅", "预测时点": "T0", "原因": "申请期限"},
    {"Feature": "Var1", "是否入模": "✅", "预测时点": "T0", "原因": "匿名脱敏特征"},
    {"Feature": "Contacted", "是否入模": "❌", "预测时点": "T1", "原因": "申请后发生，预测时点不可得"},
    {"Feature": "Interest_Rate", "是否入模": "❌", "预测时点": "T1", "原因": "审批/定价后才生成"},
    {"Feature": "EMI", "是否入模": "❌", "预测时点": "T1", "原因": "审批/定价后才生成"},
]
# ============================================================
# 特征工程
# ============================================================
class FeatureEngineer:
    """
    V1.4 特征工程处理器。
    核心原则：
    - basic_clean：仅做不依赖数据分布统计量的确定性清洗，可在时间切分前执行。
    - fit：只能对“训练部分”执行。
    - transform：将训练部分学到的统计量应用到 train / valid / test。
    """
    def __init__(self):
        self.fitted = False
        self.num_medians = {}
        self.num_upper_limits = {}
        self.age_median = np.nan
        self.cat_encoders = {}
        self.cat_cols = []
        self.loan_amount_by_period = {}
        self.loan_amount_global_median = np.nan
        self.loan_period_global_median = np.nan
        self.feature_cols = []
        self.train_columns = None
        self.fit_stats = {}
    # --------------------------------------------------------
    @staticmethod
    def basic_clean(df):
        """
        无状态清洗：
        不依赖训练集/验证集分布统计量，因此允许在切分前执行。
        """
        df = df.copy()
        # DOB: dd/mm/yy
        if "DOB" in df.columns:
            df["DOB"] = pd.to_datetime(
                df["DOB"],
                format="%d/%m/%y",
                errors="coerce"
            )
            # 两千年问题修正：例如 2068 -> 1968
            df["DOB"] = df["DOB"].apply(
                lambda dt: dt - pd.DateOffset(years=100)
                if pd.notnull(dt) and dt.year > 2020
                else dt
            )
        # 申请日期
        if "Lead_Creation_Date" in df.columns:
            df["Lead_Creation_Date"] = pd.to_datetime(
                df["Lead_Creation_Date"],
                dayfirst=True,
                errors="coerce"
            )
        # 年龄
        if "DOB" in df.columns and "Lead_Creation_Date" in df.columns:
            lead = df["Lead_Creation_Date"]
            dob = df["DOB"]
            not_yet_birthday = (
                (lead.dt.month < dob.dt.month)
                | (
                    (lead.dt.month == dob.dt.month)
                    & (lead.dt.day < dob.dt.day)
                )
            )
            df["Age"] = (
                lead.dt.year
                - dob.dt.year
                - not_yet_birthday.astype(int)
            )
            # 不合理年龄置 NaN，不硬截断
            df.loc[
                df["Age"].isna()
                | (df["Age"] < 18)
                | (df["Age"] > 65),
                "Age"
            ] = np.nan
        # 删除主键、T1字段、原始DOB
        drop_cols = [
            "ID",
            "Contacted",
            "Interest_Rate",
            "EMI",
            "DOB",
        ]
        df.drop(
            columns=[c for c in drop_cols if c in df.columns],
            inplace=True,
            errors="ignore"
        )
        # V1.4：Employer_Code 暂不进入主模型
        df.drop(
            columns=[c for c in EXCLUDED_MODEL_COLS if c in df.columns],
            inplace=True,
            errors="ignore"
        )
        # Employer_Category2 先补 UNKNOWN 再转字符串
        if "Employer_Category2" in df.columns:
            df["Employer_Category2"] = (
                df["Employer_Category2"]
                .fillna("UNKNOWN")
                .astype(str)
            )
        return df
    # --------------------------------------------------------
    def fit(self, train_df):
        """
        仅使用训练部分拟合所有统计量。
        """
        df = train_df.copy()
        # 1) 数值中位数
        self.num_medians = {
            "Monthly_Income": df["Monthly_Income"].median(),
            "Existing_EMI": df["Existing_EMI"].median(),
            "Var1": df["Var1"].median(),
        }
        # 2) Age 中位数
        self.age_median = df["Age"].median()
        # 3) 数值异常缩尾上限
        self.num_upper_limits = {
            "Monthly_Income": df["Monthly_Income"].quantile(0.99),
            "Existing_EMI": df["Existing_EMI"].quantile(0.99),
        }
        # 4) 类别列
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
        self.cat_cols = [c for c in self.cat_cols if c in df.columns]
        self.cat_encoders = {}
        for col in self.cat_cols:
            le = LabelEncoder()
            values = (
                df[col]
                .fillna("UNKNOWN")
                .astype(str)
            )
            le.fit(values.unique())
            self.cat_encoders[col] = le
        # 5) Loan 缺失填充统计量
        self.loan_amount_by_period = (
            df.groupby("Loan_Period")["Loan_Amount"]
            .median()
            .to_dict()
        )
        self.loan_amount_global_median = df["Loan_Amount"].median()
        self.loan_period_global_median = df["Loan_Period"].median()
        self.fit_stats = {
            "num_medians": self.num_medians,
            "num_upper_limits": self.num_upper_limits,
            "age_median": self.age_median,
            "loan_amount_by_period_n": len(self.loan_amount_by_period),
            "loan_amount_global_median": self.loan_amount_global_median,
            "loan_period_global_median": self.loan_period_global_median,
            "cat_n_unique": {
                c: len(self.cat_encoders[c].classes_)
                for c in self.cat_cols
            },
            "excluded_model_cols": EXCLUDED_MODEL_COLS,
        }
        self.fitted = True
        print("[fit] V1.4 统计量拟合完成（仅基于训练部分）")
        return self
    # --------------------------------------------------------
    def transform(self, df, is_train=False):
        """
        使用训练阶段已经拟合的统计量进行转换。
        """
        if not self.fitted:
            raise ValueError("请先执行 fit()")
        df = df.copy()
        # ----------------------------------------------------
        # 1. 数值填补
        # V1.4：不再生成当前数据中恒为0的三个 Missing flags
        # ----------------------------------------------------
        df["Monthly_Income"] = df["Monthly_Income"].fillna(
            self.num_medians["Monthly_Income"]
        )
        df["Existing_EMI"] = df["Existing_EMI"].fillna(
            self.num_medians["Existing_EMI"]
        )
        df["Var1"] = df["Var1"].fillna(
            self.num_medians["Var1"]
        )
        # ----------------------------------------------------
        # 2. Age
        # ----------------------------------------------------
        age_missing_mask = df["Age"].isna()
        df["Age_Missing"] = age_missing_mask.astype(int)
        df["Age"] = df["Age"].fillna(self.age_median)
        # ----------------------------------------------------
        # 3. Loan_Amount / Loan_Period
        # V1.4：合并重复缺失标记
        # ----------------------------------------------------
        loan_amount_missing = df["Loan_Amount"].isna()
        loan_period_missing = df["Loan_Period"].isna()
        df["Loan_Info_Missing"] = (
            loan_amount_missing | loan_period_missing
        ).astype(int)
        # Loan_Amount:
        # 先尝试按 Loan_Period 的训练期条件中位数填补
        amt_miss = df["Loan_Amount"].isna()
        per_exist = df["Loan_Period"].notna()
        mapped_amount = df["Loan_Period"].map(
            self.loan_amount_by_period
        )
        df.loc[
            amt_miss & per_exist,
            "Loan_Amount"
        ] = mapped_amount[amt_miss & per_exist]
        # 条件中位数仍无法填补 -> 训练期全局中位数
        df["Loan_Amount"] = df["Loan_Amount"].fillna(
            self.loan_amount_global_median
        )
        # Loan_Period -> 训练期全局中位数
        df["Loan_Period"] = df["Loan_Period"].fillna(
            self.loan_period_global_median
        )
        # ----------------------------------------------------
        # 4. 类别缺失
        # ----------------------------------------------------
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN")
        # ----------------------------------------------------
        # 5. P99 缩尾
        # ----------------------------------------------------
        df["Monthly_Income"] = df["Monthly_Income"].clip(
            upper=self.num_upper_limits["Monthly_Income"]
        )
        df["Existing_EMI"] = df["Existing_EMI"].clip(
            upper=self.num_upper_limits["Existing_EMI"]
        )
        # ----------------------------------------------------
        # 6. 业务衍生特征
        # ----------------------------------------------------
        df["EMI_Zero_Flag"] = (
            df["Existing_EMI"] == 0
        ).astype(int)
        df["DTI"] = (
            df["Existing_EMI"]
            / (df["Monthly_Income"] + 1e-5)
        ).clip(0, 5)
        df["Log_Income"] = np.log1p(
            df["Monthly_Income"]
        )
        df["Loan_to_Income"] = (
            df["Loan_Amount"]
            / (df["Monthly_Income"] + 1e-5)
        ).clip(0, 100)
        # ----------------------------------------------------
        # 7. Age_Bin
        # 原始年龄缺失时：Age 数值填中位数，但 Age_Bin 保留 -1
        # ----------------------------------------------------
        age_for_bin = df["Age"].copy()
        df["Age_Bin"] = pd.cut(
            age_for_bin,
            bins=[18, 25, 35, 45, 55, 66],
            right=False,
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
        df.loc[age_missing_mask, "Age_Bin"] = -1
        df["Age_Bin"] = df["Age_Bin"].astype(int)
        # ----------------------------------------------------
        # 8. 删除日期
        # ----------------------------------------------------
        if "Lead_Creation_Date" in df.columns:
            df.drop(
                columns=["Lead_Creation_Date"],
                inplace=True
            )
        # ----------------------------------------------------
        # 9. 类别编码
        # 训练中没见过的类别 -> -1
        # ----------------------------------------------------
        for col in self.cat_cols:
            if col in df.columns:
                le = self.cat_encoders[col]
                class_to_int = {
                    cls: i
                    for i, cls in enumerate(le.classes_)
                }
                df[col] = (
                    df[col]
                    .astype(str)
                    .map(class_to_int)
                    .fillna(-1)
                    .astype(int)
                )
        # ----------------------------------------------------
        # 10. 最终特征
        # ----------------------------------------------------
        numeric_features = [
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
        ]
        category_features = (
            self.cat_cols
            + ["Age_Bin"]
        )
        all_features = list(dict.fromkeys(
            [
                c
                for c in numeric_features + category_features
                if c in df.columns
            ]
        ))
        if "Approved" in df.columns:
            all_features = (
                [c for c in all_features if c != "Approved"]
                + ["Approved"]
            )
        df = df[all_features]
        self.feature_cols = [
            c for c in all_features
            if c != "Approved"
        ]
        if is_train:
            self.train_columns = list(self.feature_cols)
        # ----------------------------------------------------
        # 11. 列对齐
        # ----------------------------------------------------
        if not is_train and self.train_columns is not None:
            missing_cols = (
                set(self.train_columns)
                - set(df.columns)
            )
            if missing_cols:
                raise ValueError(
                    "数据缺少训练集特征列: {}".format(
                        missing_cols
                    )
                )
            extra_cols = (
                set(df.columns)
                - set(self.train_columns)
                - {"Approved"}
            )
            if extra_cols:
                df.drop(
                    columns=list(extra_cols),
                    inplace=True
                )
            if "Approved" in df.columns:
                df = df[
                    self.train_columns + ["Approved"]
                ]
            else:
                df = df[
                    self.train_columns
                ]
        return df
# ============================================================
# 工具函数
# ============================================================
def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
def load_csv_auto(path):
    """
    尽量兼容原数据常见编码。
    """
    encodings = ["gbk", "gb18030", "utf-8-sig", "utf-8"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[读取成功] {path} | encoding={enc}")
            return df
        except Exception as e:
            last_error = e
    raise last_error
def positive_summary(df, name):
    pos = int(df["Approved"].sum())
    n = len(df)
    rate = pos / n if n else np.nan
    return {
        "dataset": name,
        "rows": n,
        "positives": pos,
        "positive_rate": rate,
    }
def raw_missing_pct(frame, col):
    if col in frame.columns:
        return round(
            float(frame[col].isnull().mean() * 100),
            4
        )
    return "—"
# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    ensure_output_dir(OUTPUT_DIR)
    print("=" * 76)
    print("银行贷款审批预测：预处理与特征工程 V1.4")
    print("主验证方案：时间顺序前 80% Train / 后 20% Validation")
    print("=" * 76)
    print(f"训练集路径: {TRAIN_PATH}")
    print(f"测试集路径: {TEST_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"VALID_RATIO: {VALID_RATIO:.2f}")
    print(f"暂不入模字段: {EXCLUDED_MODEL_COLS}")
    print("=" * 76)
    # --------------------------------------------------------
    # 1. 读取原始数据
    # --------------------------------------------------------
    train_raw = load_csv_auto(TRAIN_PATH)
    test_raw = load_csv_auto(TEST_PATH)
    print(
        f"\n原始数据："
        f"train={train_raw.shape}, "
        f"test={test_raw.shape}"
    )
    if "Approved" not in train_raw.columns:
        raise ValueError("train.csv 中未找到 Approved")
    if "ID" not in test_raw.columns:
        raise ValueError("test_features.csv 中未找到 ID")
    test_id = test_raw["ID"].copy()
    # --------------------------------------------------------
    # 2. 字段审计表
    # --------------------------------------------------------
    field_audit = pd.DataFrame(
        FIELD_AVAILABILITY
    )
    audit_path = os.path.join(
        OUTPUT_DIR,
        "field_availability_audit_v14.csv"
    )
    field_audit.to_csv(
        audit_path,
        index=False,
        encoding=OUT_ENCODING
    )
    print(
        "\n✅ 已保存字段审计表："
        "field_availability_audit_v14.csv"
    )
    # --------------------------------------------------------
    # 3. 无状态清洗
    # --------------------------------------------------------
    train_clean = FeatureEngineer.basic_clean(
        train_raw
    )
    test_clean = FeatureEngineer.basic_clean(
        test_raw
    )
    # --------------------------------------------------------
    # 4. 时间排序
    # --------------------------------------------------------
    if train_clean["Lead_Creation_Date"].isna().any():
        bad_date_n = int(
            train_clean["Lead_Creation_Date"]
            .isna()
            .sum()
        )
        print(
            f"\n⚠️ Lead_Creation_Date 有 "
            f"{bad_date_n} 条无法解析。"
        )
    train_clean = (
        train_clean
        .sort_values(
            "Lead_Creation_Date",
            kind="mergesort"
        )
        .reset_index(drop=True)
    )
    # --------------------------------------------------------
    # 5. 时间顺序 80/20 切分
    # --------------------------------------------------------
    if not 0 < VALID_RATIO < 1:
        raise ValueError(
            "VALID_RATIO 必须在 (0, 1) 之间"
        )
    split_idx = int(
        len(train_clean)
        * (1 - VALID_RATIO)
    )
    train_df = (
        train_clean
        .iloc[:split_idx]
        .copy()
    )
    valid_df = (
        train_clean
        .iloc[split_idx:]
        .copy()
    )
    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError(
            "时间切分后训练集或验证集为空"
        )
    # --------------------------------------------------------
    # 6. 切分结果
    # --------------------------------------------------------
    print("\n[时间切分结果]")
    split_rows = []
    for name, d in [
        ("train", train_df),
        ("valid", valid_df)
    ]:
        pos = int(d["Approved"].sum())
        rate = pos / len(d)
        min_date = d[
            "Lead_Creation_Date"
        ].min()
        max_date = d[
            "Lead_Creation_Date"
        ].max()
        print(
            f"  {name:<6} "
            f"| {len(d):>6} 行 "
            f"| 正样本 {pos:>4} "
            f"| 正样本率 {rate * 100:.4f}% "
            f"| {min_date.date()} ~ {max_date.date()}"
        )
        split_rows.append({
            "dataset": name,
            "rows": len(d),
            "positives": pos,
            "positive_rate_pct": round(
                rate * 100, 6
            ),
            "date_min": str(
                min_date.date()
            ) if pd.notnull(min_date) else "",
            "date_max": str(
                max_date.date()
            ) if pd.notnull(max_date) else "",
        })
    vp = int(
        valid_df["Approved"].sum()
    )
    if vp < MIN_VALID_POSITIVES:
        print(
            f"⚠️ 验证集正样本仅 {vp} 个，"
            f"低于 {MIN_VALID_POSITIVES}"
        )
    split_summary = pd.DataFrame(
        split_rows
    )
    split_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "split_summary_v14.csv"
        ),
        index=False,
        encoding=OUT_ENCODING
    )
    print(
        "✅ 已保存 split_summary_v14.csv"
    )
    # --------------------------------------------------------
    # 7. 贷款缺失模式审计
    # --------------------------------------------------------
    amount_mask = (
        train_df["Loan_Amount"].isna()
    )
    period_mask = (
        train_df["Loan_Period"].isna()
    )
    same_mask_rate = (
        amount_mask == period_mask
    ).mean()
    print(
        "\n[Loan 缺失模式审计]"
    )
    print(
        "  Loan_Amount 与 Loan_Period "
        f"缺失标记一致率：{same_mask_rate * 100:.4f}%"
    )
    print(
        "  V1.4 使用统一特征：Loan_Info_Missing"
    )
    # --------------------------------------------------------
    # 8. City_Code -> City_Category 审计
    # --------------------------------------------------------
    if (
        "City_Code" in train_df.columns
        and "City_Category" in train_df.columns
    ):
        cc = (
            train_df
            .dropna(
                subset=[
                    "City_Code",
                    "City_Category"
                ]
            )
            .groupby(
                "City_Code"
            )["City_Category"]
            .nunique()
        )
        multi = int(
            (cc > 1).sum()
        )
        print(
            "\n[City_Code -> City_Category 审计]"
        )
        print(
            f"  唯一 City_Code 数: {len(cc)}"
        )
        print(
            f"  一对多 (>1类): {multi}"
        )
    # --------------------------------------------------------
    # 9. Fit：只用训练部分
    # --------------------------------------------------------
    fe = FeatureEngineer()
    fe.fit(train_df)
    # --------------------------------------------------------
    # 10. Transform
    # --------------------------------------------------------
    train_processed = fe.transform(
        train_df,
        is_train=True
    )
    valid_processed = fe.transform(
        valid_df,
        is_train=False
    )
    test_processed = fe.transform(
        test_clean,
        is_train=False
    )
    # 标签拆出再落盘
    y_train = (
        train_processed["Approved"]
        .copy()
    )
    y_valid = (
        valid_processed["Approved"]
        .copy()
    )
    X_train = train_processed.drop(
        columns=["Approved"]
    )
    X_valid = valid_processed.drop(
        columns=["Approved"]
    )
    X_test = test_processed.copy()
    # --------------------------------------------------------
    # 11. 最终完整性检查
    # --------------------------------------------------------
    print("\n[最终完整性检查]")
    print(
        f"  X_train: {X_train.shape}"
    )
    print(
        f"  X_valid: {X_valid.shape}"
    )
    print(
        f"  X_test : {X_test.shape}"
    )
    print(
        f"  特征列数量: {len(fe.feature_cols)}"
    )
    train_nan = int(
        X_train.isna().sum().sum()
    )
    valid_nan = int(
        X_valid.isna().sum().sum()
    )
    test_nan = int(
        X_test.isna().sum().sum()
    )
    print(
        f"  NaN 总数 -> "
        f"train={train_nan}, "
        f"valid={valid_nan}, "
        f"test={test_nan}"
    )
    # 检查 inf
    def inf_count(frame):
        arr = frame.select_dtypes(
            include=[np.number]
        ).to_numpy()
        return int(
            np.isinf(arr).sum()
        )
    print(
        f"  Inf 总数 -> "
        f"train={inf_count(X_train)}, "
        f"valid={inf_count(X_valid)}, "
        f"test={inf_count(X_test)}"
    )
    if list(X_train.columns) != list(X_valid.columns):
        raise ValueError(
            "Train / Valid 特征列顺序不一致"
        )
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Train / Test 特征列顺序不一致"
        )
    # --------------------------------------------------------
    # 12. 未见类别比例
    # --------------------------------------------------------
    unseen_rows = []
    print("\n[类别未见值比例]")
    for col in fe.cat_cols:
        valid_unseen = float(
            (X_valid[col] == -1).mean()
        )
        test_unseen = float(
            (X_test[col] == -1).mean()
        )
        unseen_rows.append({
            "feature": col,
            "valid_unseen_pct": round(
                valid_unseen * 100, 4
            ),
            "test_unseen_pct": round(
                test_unseen * 100, 4
            ),
        })
        print(
            f"  {col:<45} "
            f"valid={valid_unseen * 100:>8.4f}% "
            f"test={test_unseen * 100:>8.4f}%"
        )
    unseen_df = pd.DataFrame(
        unseen_rows
    )
    unseen_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "category_unseen_report_v14.csv"
        ),
        index=False,
        encoding=OUT_ENCODING
    )
    print(
        "✅ 已保存 category_unseen_report_v14.csv"
    )
    # --------------------------------------------------------
    # 13. 预处理审计报告
    # --------------------------------------------------------
    rows = []
    for col in fe.feature_cols:
        is_cat = (
            col in fe.cat_cols
            or col == "Age_Bin"
        )
        fill_method = "—"
        clip_note = "—"
        valid_unseen = "—"
        test_unseen = "—"
        if col in fe.cat_cols:
            fill_method = (
                "'UNKNOWN'，训练未见类别 -> -1"
            )
            valid_unseen = round(
                float(
                    (X_valid[col] == -1)
                    .mean()
                    * 100
                ),
                4
            )
            test_unseen = round(
                float(
                    (X_test[col] == -1)
                    .mean()
                    * 100
                ),
                4
            )
        if col == "Monthly_Income":
            fill_method = "训练中位数"
            clip_note = "训练P99上尾"
        elif col == "Existing_EMI":
            fill_method = "训练中位数"
            clip_note = "训练P99上尾"
        elif col == "Var1":
            fill_method = "训练中位数"
        elif col == "Loan_Amount":
            fill_method = (
                "按期限条件中位数 → "
                "训练全局中位数"
            )
        elif col == "Loan_Period":
            fill_method = "训练全局中位数"
        elif col == "Loan_Info_Missing":
            fill_method = (
                "Loan_Amount 或 Loan_Period "
                "原始缺失 -> 1"
            )
        elif col == "Age":
            fill_method = "训练 Age 中位数"
        elif col == "Age_Missing":
            fill_method = "原始 Age 缺失 -> 1"
        elif col == "Age_Bin":
            fill_method = (
                "年龄分箱；原始 Age 缺失 -> -1"
            )
        elif col == "DTI":
            fill_method = "衍生"
            clip_note = "clip[0,5]"
        elif col == "Log_Income":
            fill_method = "log1p衍生"
        elif col == "Loan_to_Income":
            fill_method = "衍生"
            clip_note = "clip[0,100]"
        elif col == "EMI_Zero_Flag":
            fill_method = "Existing_EMI==0 -> 1"
        rows.append({
            "列": col,
            "类型": (
                "类别"
                if is_cat
                else "数值"
            ),
            "训练原始缺失%": raw_missing_pct(
                train_df,
                col
            ),
            "验证原始缺失%": raw_missing_pct(
                valid_df,
                col
            ),
            "测试原始缺失%": raw_missing_pct(
                test_clean,
                col
            ),
            "填充/处理": fill_method,
            "拟合来源": "仅训练部分",
            "缩尾/约束": clip_note,
            "验证未见过%": valid_unseen,
            "测试未见过%": test_unseen,
        })
    report = pd.DataFrame(
        rows
    )
    report.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "preprocess_report_v14.csv"
        ),
        index=False,
        encoding=OUT_ENCODING
    )
    print(
        "✅ 已保存 preprocess_report_v14.csv"
    )
    # --------------------------------------------------------
    # 14. 落盘：train / valid / test
    # --------------------------------------------------------
    train_out = X_train.copy()
    train_out["Approved"] = y_train.values
    valid_out = X_valid.copy()
    valid_out["Approved"] = y_valid.values
    test_out = X_test.copy()
    test_out.insert(
        0,
        "ID",
        test_id.values
    )
    train_file = os.path.join(
        OUTPUT_DIR,
        "train_processed_v14.csv"
    )
    valid_file = os.path.join(
        OUTPUT_DIR,
        "valid_processed_v14.csv"
    )
    test_file = os.path.join(
        OUTPUT_DIR,
        "test_processed_v14.csv"
    )
    train_out.to_csv(
        train_file,
        index=False,
        encoding=OUT_ENCODING
    )
    valid_out.to_csv(
        valid_file,
        index=False,
        encoding=OUT_ENCODING
    )
    test_out.to_csv(
        test_file,
        index=False,
        encoding=OUT_ENCODING
    )
    print(
        "\n✅ 已保存："
        "train_processed_v14.csv"
    )
    print(
        "✅ 已保存："
        "valid_processed_v14.csv"
    )
    print(
        "✅ 已保存："
        "test_processed_v14.csv"
    )
    # --------------------------------------------------------
    # 15. 类别特征清单
    # --------------------------------------------------------
    category_file = os.path.join(
        OUTPUT_DIR,
        "categorical_features_v14.txt"
    )
    with open(
        category_file,
        "w",
        encoding=OUT_ENCODING
    ) as f:
        for c in fe.cat_cols + ["Age_Bin"]:
            f.write(c + "\n")
    print(
        "✅ 已保存 categorical_features_v14.txt"
    )
    # --------------------------------------------------------
    # 16. 特征列表
    # --------------------------------------------------------
    feature_file = os.path.join(
        OUTPUT_DIR,
        "feature_list_v14.txt"
    )
    with open(
        feature_file,
        "w",
        encoding=OUT_ENCODING
    ) as f:
        for c in fe.feature_cols:
            f.write(c + "\n")
    print(
        "✅ 已保存 feature_list_v14.txt"
    )
    # --------------------------------------------------------
    # 17. Fit statistics
    # --------------------------------------------------------
    fit_stats_file = os.path.join(
        OUTPUT_DIR,
        "fit_stats_v14.json"
    )
    # numpy 类型转 Python 原生类型，便于 JSON 保存
    serializable_fit_stats = json.loads(
        json.dumps(
            fe.fit_stats,
            default=lambda x: (
                float(x)
                if isinstance(
                    x,
                    (
                        np.floating,
                        np.integer
                    )
                )
                else str(x)
            ),
            ensure_ascii=False
        )
    )
    with open(
        fit_stats_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            serializable_fit_stats,
            f,
            ensure_ascii=False,
            indent=2
        )
    print(
        "✅ 已保存 fit_stats_v14.json"
    )
    # --------------------------------------------------------
    # 18. 运行摘要
    # --------------------------------------------------------
    summary_file = os.path.join(
        OUTPUT_DIR,
        "run_summary_v14.txt"
    )
    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "银行贷款审批预测 - "
            "预处理 V1.4 运行摘要\n"
        )
        f.write("=" * 60 + "\n")
        f.write(
            f"原始训练集: {train_raw.shape}\n"
        )
        f.write(
            f"原始测试集: {test_raw.shape}\n"
        )
        f.write(
            f"时间切分: 前 {1 - VALID_RATIO:.0%} "
            f"Train / 后 {VALID_RATIO:.0%} Valid\n"
        )
        f.write(
            f"训练部分: {len(train_df)} 行, "
            f"正样本 {int(train_df['Approved'].sum())}, "
            f"正样本率 "
            f"{train_df['Approved'].mean() * 100:.6f}%\n"
        )
        f.write(
            f"验证部分: {len(valid_df)} 行, "
            f"正样本 {int(valid_df['Approved'].sum())}, "
            f"正样本率 "
            f"{valid_df['Approved'].mean() * 100:.6f}%\n"
        )
        f.write(
            f"最终特征数: {len(fe.feature_cols)}\n"
        )
        f.write(
            "V1.4 暂不入模字段: "
            + ", ".join(EXCLUDED_MODEL_COLS)
            + "\n"
        )
        f.write(
            f"Age 训练中位数: {fe.age_median}\n"
        )
        f.write(
            f"最终 NaN 总数: "
            f"train={train_nan}, "
            f"valid={valid_nan}, "
            f"test={test_nan}\n"
        )
        f.write(
            "最终特征列表:\n"
        )
        for c in fe.feature_cols:
            f.write(
                f"  - {c}\n"
            )
    print(
        "✅ 已保存 run_summary_v14.txt"
    )
    # --------------------------------------------------------
    # 19. 最终输出
    # --------------------------------------------------------
    print("\n" + "=" * 76)
    print(
        "V1.4 预处理完成。"
    )
    print(
        f"最终特征共 {len(fe.feature_cols)} 个："
    )
    print(fe.feature_cols)
    print("=" * 76)
    print(
        "\n下一步请把以下文件发给我："
    )
    print(
        "1) train_processed_v14.csv"
    )
    print(
        "2) valid_processed_v14.csv"
    )
    print(
        "3) test_processed_v14.csv"
    )
    print(
        "4) preprocess_report_v14.csv"
    )
    print(
        "5) split_summary_v14.csv"
    )
    print(
        "6) category_unseen_report_v14.csv"
    )
    print(
        "7) run_summary_v14.txt"
    )
