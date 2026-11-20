# -*- coding: utf-8 -*-
"""
RF + LSTM 水温缺失值填补统一比较程序（审稿意见修订版）

主要改进：
1. 正式研究时段严格限定为 2023-06-01 至 2025-05-31，共两个水文年。
2. 2023-01-01 至 2023-05-31 仅作为特征预热期，不参与正式样本量、缺失率和性能统计。
3. 2025-06-01 之后的数据不参与正式训练、验证、模型选择或填补结果。
4. 使用连续时间块模拟真实缺口，替代随机训练/测试拆分。
5. RF和LSTM使用完全相同的验证时间块，保证公平比较。
6. 所有预处理参数只在各折训练集上拟合，避免时间泄漏。
7. 移除居中滑动平均、未来数据回填、RF人为随机扰动等不合理处理。
8. 风向WD按原始弧度直接计算正余弦，不再使用np.radians重复转换。
9. 使用按缺口长度分组的交叉保形校准，生成95%预测区间。
10. 自动选择RF或LSTM中表现更好的模型，并生成多重插补序列。
11. 图中文字使用英文Arial；代码注释、控制台提示和说明报告使用中文。

说明：
- 本程序首先针对水温WT进行填补。
- 后续水质变量可复用同一验证、区间校准和模型选择框架。
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "本程序需要PyTorch。请先运行：pip install torch"
    ) from exc

warnings.filterwarnings("ignore")


# ============================================================
# 1. 用户参数区
# ============================================================

# 原始TIN文件路径。请按本机实际情况修改。
INPUT_FILE = str(Path(__file__).resolve().parents[1]/"data/TIN_2023-2024-2025.csv")

# 输出目录。请按本机实际情况修改。
OUTPUT_DIR = str(Path(__file__).resolve().parents[2]/"generated/temperature")

# JDY=1对应的真实日期
BASE_DATE = pd.Timestamp("2023-01-01")

# 特征预热期：只提供气象、流量等历史上下文，不计入正式结果
WARMUP_START = pd.Timestamp("2023-01-01")

# 论文正式研究期：两个水文年
STUDY_START = pd.Timestamp("2023-06-01")
STUDY_END = pd.Timestamp("2025-05-31")

# 物理合理范围，仅用于最终异常边界检查，不进行人为平滑或随机扰动
PHYSICAL_MIN_WT = 0.0
PHYSICAL_MAX_WT = 35.0

# 验证设置：连续时间块长度，覆盖短缺口到超长缺口
VALIDATION_BLOCK_LENGTHS = [3, 7, 15, 30, 60, 90, 150]
BLOCKS_PER_LENGTH = 3
RANDOM_SEED = 42

# 多重插补次数，用于后续不确定性传播
N_MULTIPLE_IMPUTATIONS = 100

# 是否执行LSTM。没有GPU时运行时间会明显长于RF。
RUN_LSTM = True

# 快速测试模式：用于先检查代码能否完整运行。
# 正式论文结果必须设为False。
QUICK_MODE = False

# 图件设置
FIGURE_DPI = 600

# RF候选参数。正式程序通过相同连续时间块验证选择最优组合。
RF_CANDIDATES = [
    {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 3,
        "min_samples_split": 6,
        "max_features": 0.7,
        "bootstrap": True,
    },
    {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 3,
        "min_samples_split": 6,
        "max_features": 0.8,
        "bootstrap": True,
    },
    {
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": 0.8,
        "bootstrap": True,
    },
    {
        "n_estimators": 400,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "min_samples_split": 5,
        "max_features": "sqrt",
        "bootstrap": True,
    },
]


@dataclass(frozen=True)
class LSTMConfig:
    sequence_length: int
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 140
    patience: int = 18


LSTM_CANDIDATES = [
    LSTMConfig(sequence_length=7, hidden_size=32, num_layers=1, dropout=0.15),
    LSTMConfig(sequence_length=14, hidden_size=32, num_layers=1, dropout=0.15),
    LSTMConfig(sequence_length=21, hidden_size=32, num_layers=1, dropout=0.20),
]

# 最终LSTM使用不同随机种子形成深度集成。
LSTM_ENSEMBLE_SEEDS = [42, 52, 62]


# ============================================================
# 2. 全局绘图与随机种子
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 限制CPU线程数，避免PyTorch在多核机器上因线程过多而显著变慢。
# 使用GPU时该设置只影响CPU侧预处理。
TORCH_NUM_THREADS = max(1, min(4, os.cpu_count() or 1))
torch.set_num_threads(TORCH_NUM_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def set_all_seeds(seed: int) -> None:
    """固定Python、NumPy和PyTorch随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_all_seeds(RANDOM_SEED)


# ============================================================
# 3. 基础工具函数
# ============================================================


def ensure_dir(path: str | Path) -> Path:
    """创建目录并返回Path对象。"""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def gap_category(length: int) -> str:
    """按照连续缺失长度划分类别。"""
    if length <= 3:
        return "1-3 days"
    if length <= 15:
        return "4-15 days"
    if length <= 60:
        return "16-60 days"
    return ">60 days"


def finite_sample_quantile(values: Sequence[float], confidence: float = 0.95) -> float:
    """计算保形预测区间使用的有限样本高分位数。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    level = math.ceil((len(arr) + 1) * confidence) / len(arr)
    level = min(level, 1.0)
    try:
        return float(np.quantile(arr, level, method="higher"))
    except TypeError:
        return float(np.quantile(arr, level, interpolation="higher"))


def safe_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """样本量或方差不足时返回NaN，避免R²报错。"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) < 2 or np.nanstd(y_true_arr) == 0:
        return np.nan
    return float(r2_score(y_true_arr, y_pred_arr))


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """计算统一的回归评价指标。"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]

    if len(y_true_arr) == 0:
        return {
            "N": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "R2": np.nan,
        }

    residual = y_pred_arr - y_true_arr
    return {
        "N": int(len(y_true_arr)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "MAE": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "Bias": float(np.mean(residual)),
        "R2": safe_r2(y_true_arr, y_pred_arr),
    }


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    """保存600 dpi PNG和可编辑PDF。"""
    fig.savefig(str(output_base) + ".png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 4. 数据读取、日期处理和特征工程
# ============================================================


def load_and_prepare_data(file_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取TIN文件并生成日期。

    返回：
    1. context_df：预热期开始至研究期结束，用于生成滞后和序列特征。
    2. study_df：仅研究期731天，用于正式统计、验证和输出。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"未找到输入文件：{file_path}")

    raw = pd.read_csv(file_path)
    raw.columns = [str(c).strip() for c in raw.columns]

    required = ["JDY", "WT", "QIN", "TAIR", "TDEW", "WS", "WD", "CLOUD"]
    missing_columns = [c for c in required if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"输入文件缺少必要列：{missing_columns}")

    df = raw[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = BASE_DATE + pd.to_timedelta(df["JDY"] - 1, unit="D")
    df = df.sort_values("Date").drop_duplicates("Date", keep="first").reset_index(drop=True)

    # 日期完整性检查
    expected_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    if len(expected_dates) != len(df) or not np.array_equal(expected_dates.values, df["Date"].values):
        raise ValueError("日期序列不连续或存在重复，请先检查JDY。")

    # 预测因子完整性检查
    predictor_cols = ["QIN", "TAIR", "TDEW", "WS", "WD", "CLOUD"]
    predictor_missing = df[predictor_cols].isna().sum()
    if predictor_missing.sum() > 0:
        print("警告：预测因子存在缺失值。程序会在每个训练折内使用中位数填补：")
        print(predictor_missing[predictor_missing > 0])

    # 删除研究期之后的数据；保留研究期之前的预热数据
    context_df = df[(df["Date"] >= WARMUP_START) & (df["Date"] <= STUDY_END)].copy()
    context_df = context_df.reset_index(drop=True)
    context_df["Context_Row"] = np.arange(len(context_df), dtype=int)
    context_df["In_Study_Period"] = (
        (context_df["Date"] >= STUDY_START) & (context_df["Date"] <= STUDY_END)
    )

    study_df = context_df[context_df["In_Study_Period"]].copy().reset_index(drop=True)
    study_df["Study_Pos"] = np.arange(len(study_df), dtype=int)

    expected_study_dates = pd.date_range(STUDY_START, STUDY_END, freq="D")
    if len(study_df) != len(expected_study_dates):
        raise ValueError(
            f"研究期应有{len(expected_study_dates)}天，但读取到{len(study_df)}天。"
        )

    print("\n数据读取完成")
    print(f"原始日期范围：{df['Date'].min().date()} 至 {df['Date'].max().date()}")
    print(f"特征预热期：{WARMUP_START.date()} 至 {(STUDY_START - pd.Timedelta(days=1)).date()}")
    print(f"正式研究期：{STUDY_START.date()} 至 {STUDY_END.date()}，共{len(study_df)}天")
    print(
        f"研究期水温实测{study_df['WT'].notna().sum()}天，缺失{study_df['WT'].isna().sum()}天，"
        f"缺失率{study_df['WT'].isna().mean() * 100:.2f}%"
    )

    return context_df, study_df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成不受闰年相位漂移影响的时间特征。"""
    out = df.copy()
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["DayOfYear"] = out["Date"].dt.dayofyear
    out["DaysInYear"] = np.where(out["Date"].dt.is_leap_year, 366.0, 365.0)
    annual_phase = 2.0 * np.pi * (out["DayOfYear"] - 1) / out["DaysInYear"]
    out["Annual_sin"] = np.sin(annual_phase)
    out["Annual_cos"] = np.cos(annual_phase)

    month_phase = 2.0 * np.pi * (out["Month"] - 1) / 12.0
    out["Month_sin"] = np.sin(month_phase)
    out["Month_cos"] = np.cos(month_phase)

    out["Time_Index"] = (out["Date"] - STUDY_START).dt.days.astype(float)

    # 原始WD已经是弧度，直接计算正余弦
    out["WD_sin"] = np.sin(out["WD"])
    out["WD_cos"] = np.cos(out["WD"])
    return out


def create_rf_features(context_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """创建仅使用当日及过去信息的RF特征。"""
    df = add_calendar_features(context_df)
    predictors = ["QIN", "TAIR", "TDEW", "WS", "CLOUD"]

    # 只使用过去时刻的滞后特征
    for col in predictors:
        for lag in [1, 3, 7, 14]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # 移动统计包含当日和过去，不使用center=True
    for col in predictors:
        for window in [3, 7, 14]:
            df[f"{col}_mean_{window}"] = df[col].rolling(
                window=window, min_periods=1
            ).mean()
            df[f"{col}_std_{window}"] = df[col].rolling(
                window=window, min_periods=2
            ).std()

    # 变化率和物理相关交互项
    for col in ["QIN", "TAIR", "TDEW", "WS"]:
        df[f"{col}_diff_1"] = df[col].diff(1)
        df[f"{col}_diff_3"] = df[col].diff(3)

    df["TAIR_TDEW_diff"] = df["TAIR"] - df["TDEW"]
    df["TAIR_QIN_interaction"] = df["TAIR"] * df["QIN"]

    feature_columns = [
        "Time_Index",
        "Annual_sin",
        "Annual_cos",
        "Month_sin",
        "Month_cos",
        "QIN",
        "TAIR",
        "TDEW",
        "WS",
        "CLOUD",
        "WD_sin",
        "WD_cos",
        "TAIR_TDEW_diff",
        "TAIR_QIN_interaction",
    ]

    generated = [
        c
        for c in df.columns
        if any(
            token in c
            for token in ["_lag_", "_mean_", "_std_", "_diff_"]
        )
    ]
    feature_columns.extend(generated)
    feature_columns = list(dict.fromkeys(feature_columns))
    return df, feature_columns


def create_lstm_features(context_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """创建LSTM序列输入特征，不使用目标WT的滞后值。"""
    df = add_calendar_features(context_df)
    feature_columns = [
        "Time_Index",
        "Annual_sin",
        "Annual_cos",
        "Month_sin",
        "Month_cos",
        "QIN",
        "TAIR",
        "TDEW",
        "WS",
        "CLOUD",
        "WD_sin",
        "WD_cos",
    ]
    return df, feature_columns


def assign_actual_gap_information(study_df: pd.DataFrame) -> pd.DataFrame:
    """为研究期真实缺失段分配编号、长度和类别。"""
    out = study_df.copy()
    missing = out["WT"].isna().to_numpy()
    gap_ids = np.zeros(len(out), dtype=int)
    gap_lengths = np.zeros(len(out), dtype=int)
    gap_categories = np.array(["Observed"] * len(out), dtype=object)

    gap_id = 0
    start = None
    for i, flag in enumerate(missing):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(missing) - 1):
            end = i - 1 if not flag else i
            gap_id += 1
            length = end - start + 1
            gap_ids[start : end + 1] = gap_id
            gap_lengths[start : end + 1] = length
            gap_categories[start : end + 1] = gap_category(length)
            start = None

    out["Gap_ID"] = gap_ids
    out["Gap_Length"] = gap_lengths
    out["Gap_Category"] = gap_categories
    return out


# ============================================================
# 5. 连续时间块验证设计
# ============================================================


def minimum_eval_count(block_length: int) -> int:
    """不同长度验证块所需的最低实测评价点数。"""
    if block_length <= 3:
        return 1
    if block_length <= 15:
        return 3
    if block_length <= 60:
        return 8
    return 15


def generate_validation_blocks(
    study_df: pd.DataFrame,
    block_lengths: Sequence[int],
    blocks_per_length: int,
    random_seed: int,
) -> List[Dict[str, object]]:
    """
    在研究期内选择连续时间块，并划分为调参、区间校准和独立测试三类。

    正式模式建议每种长度至少3个时间块：
    - 第1个：Tuning，用于RF参数和LSTM结构选择；
    - 第2个：Calibration，用于RF与LSTM最终选择及95%预测区间校准；
    - 第3个及以后：Test，用于独立性能报告。

    验证时会暂时隐藏时间块内所有实测WT，模型只能使用时间块外的WT进行训练，
    但允许使用时间块内完整的气象和流量预测因子，符合回顾性数据填补场景。
    """
    rng = np.random.default_rng(random_seed)
    observed = study_df["WT"].notna().to_numpy()
    n = len(study_df)
    blocks: List[Dict[str, object]] = []

    for length in block_lengths:
        candidates = []
        for start in range(1, n - length):
            end = start + length - 1
            eval_count = int(observed[start : end + 1].sum())
            train_count = int(observed.sum() - eval_count)
            if eval_count >= minimum_eval_count(length) and train_count >= 150:
                candidates.append(start)

        if not candidates:
            print(f"警告：长度{length}天没有满足条件的验证块，已跳过。")
            continue

        candidates_arr = np.asarray(candidates, dtype=int)
        rng.shuffle(candidates_arr)

        selected: List[int] = []
        min_separation = max(1, length // 2)
        for start in candidates_arr:
            if all(abs(start - s0) >= min_separation for s0 in selected):
                selected.append(int(start))
            if len(selected) >= blocks_per_length:
                break

        if len(selected) < blocks_per_length:
            for start in candidates_arr:
                if int(start) not in selected:
                    selected.append(int(start))
                if len(selected) >= blocks_per_length:
                    break

        for rep, start in enumerate(sorted(selected), start=1):
            end = start + length - 1
            eval_positions = np.where(observed[start : end + 1])[0] + start
            if blocks_per_length >= 3:
                role = "Tuning" if rep == 1 else ("Calibration" if rep == 2 else "Test")
            elif blocks_per_length == 2:
                role = "Tuning" if rep == 1 else "CalibrationTest"
            else:
                role = "QuickCheck"

            blocks.append(
                {
                    "Block_ID": f"L{length:03d}_R{rep:02d}",
                    "Validation_Role": role,
                    "Block_Length": int(length),
                    "Gap_Category": gap_category(int(length)),
                    "Start_Pos": int(start),
                    "End_Pos": int(end),
                    "Start_Date": study_df.loc[start, "Date"],
                    "End_Date": study_df.loc[end, "Date"],
                    "Eval_Positions": eval_positions.astype(int),
                    "Eval_Count": int(len(eval_positions)),
                }
            )

    if not blocks:
        raise RuntimeError("没有成功生成任何连续时间块验证样本。")

    print("\n连续时间块验证设计：")
    for block in blocks:
        print(
            f"  {block['Block_ID']} [{block['Validation_Role']}]："
            f"{block['Start_Date'].date()} 至 {block['End_Date'].date()}，"
            f"长度{block['Block_Length']}天，可评价实测点{block['Eval_Count']}个"
        )
    return blocks

def blocks_to_dataframe(blocks: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """将验证块信息转换成可保存的表格。"""
    rows = []
    for b in blocks:
        rows.append(
            {
                "验证块编号": b["Block_ID"],
                "验证用途": b.get("Validation_Role", "未标记"),
                "模拟缺口长度（天）": b["Block_Length"],
                "缺口类别": b["Gap_Category"],
                "开始日期": pd.Timestamp(b["Start_Date"]).strftime("%Y-%m-%d"),
                "结束日期": pd.Timestamp(b["End_Date"]).strftime("%Y-%m-%d"),
                "评价实测点数": b["Eval_Count"],
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 6. 线性插值和PCHIP基线验证
# ============================================================


def interpolation_prediction(
    study_df: pd.DataFrame,
    block: Dict[str, object],
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在指定连续时间块上验证插值基线。"""
    original = study_df["WT"].to_numpy(dtype=float)
    masked = original.copy()
    start = int(block["Start_Pos"])
    end = int(block["End_Pos"])
    masked[start : end + 1] = np.nan

    x = np.arange(len(masked), dtype=float)
    observed_mask = np.isfinite(masked)
    eval_positions = np.asarray(block["Eval_Positions"], dtype=int)

    if method == "Linear":
        filled = pd.Series(masked).interpolate(
            method="linear", limit_direction="both"
        ).to_numpy(dtype=float)
    elif method == "PCHIP":
        if observed_mask.sum() < 4:
            filled = pd.Series(masked).interpolate(
                method="linear", limit_direction="both"
            ).to_numpy(dtype=float)
        else:
            interpolator = PchipInterpolator(
                x[observed_mask], masked[observed_mask], extrapolate=False
            )
            filled = interpolator(x)
            # 极少数边界NaN退回线性插值
            linear = pd.Series(masked).interpolate(
                method="linear", limit_direction="both"
            ).to_numpy(dtype=float)
            filled = np.where(np.isfinite(filled), filled, linear)
    else:
        raise ValueError(f"不支持的插值方法：{method}")

    return eval_positions, original[eval_positions], filled[eval_positions]


def validate_interpolation_baselines(
    study_df: pd.DataFrame,
    blocks: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    """对线性插值和PCHIP进行与机器学习完全相同的连续块验证。"""
    rows = []
    for method in ["Linear", "PCHIP"]:
        for block in blocks:
            positions, y_true, y_pred = interpolation_prediction(study_df, block, method)
            for pos, true, pred in zip(positions, y_true, y_pred):
                rows.append(
                    {
                        "Model": method,
                        "Block_ID": block["Block_ID"],
                        "Block_Length": block["Block_Length"],
                        "Gap_Category": block["Gap_Category"],
                        "Study_Pos": int(pos),
                        "Date": study_df.loc[int(pos), "Date"],
                        "Observed": float(true),
                        "Predicted": float(pred),
                        "Residual": float(pred - true),
                    }
                )
    return pd.DataFrame(rows)


# ============================================================
# 7. 随机森林模型
# ============================================================


def rf_model_score(validation_df: pd.DataFrame) -> float:
    """用于RF内部参数选择的综合误差分数。"""
    overall = regression_metrics(validation_df["Observed"], validation_df["Predicted"])
    long_df = validation_df[validation_df["Gap_Category"].isin(["16-60 days", ">60 days"])]
    long_metrics = regression_metrics(long_df["Observed"], long_df["Predicted"])
    long_rmse = long_metrics["RMSE"] if np.isfinite(long_metrics["RMSE"]) else overall["RMSE"]
    return float(
        0.45 * overall["RMSE"]
        + 0.20 * overall["MAE"]
        + 0.25 * long_rmse
        + 0.10 * abs(overall["Bias"])
    )


def validate_rf_candidate(
    rf_df: pd.DataFrame,
    study_df: pd.DataFrame,
    feature_columns: Sequence[str],
    blocks: Sequence[Dict[str, object]],
    params: Dict[str, object],
    candidate_id: int,
) -> pd.DataFrame:
    """使用连续时间块验证一个RF参数组合。"""
    study_context_rows = study_df["Context_Row"].to_numpy(dtype=int)
    all_observed = study_df["WT"].notna().to_numpy()
    rows = []

    for block_index, block in enumerate(blocks):
        in_block = np.zeros(len(study_df), dtype=bool)
        in_block[int(block["Start_Pos"]) : int(block["End_Pos"]) + 1] = True
        train_mask = all_observed & (~in_block)
        eval_positions = np.asarray(block["Eval_Positions"], dtype=int)

        train_context_rows = study_context_rows[train_mask]
        eval_context_rows = study_context_rows[eval_positions]

        X_train_raw = rf_df.loc[train_context_rows, feature_columns].to_numpy(dtype=float)
        y_train = study_df.loc[train_mask, "WT"].to_numpy(dtype=float)
        X_eval_raw = rf_df.loc[eval_context_rows, feature_columns].to_numpy(dtype=float)
        y_eval = study_df.loc[eval_positions, "WT"].to_numpy(dtype=float)

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train_raw)
        X_eval = imputer.transform(X_eval_raw)

        model = RandomForestRegressor(
            **params,
            random_state=RANDOM_SEED + candidate_id * 100 + block_index,
            n_jobs=-1,
            oob_score=False,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_eval)
        pred = np.clip(pred, PHYSICAL_MIN_WT, PHYSICAL_MAX_WT)

        for pos, true, value in zip(eval_positions, y_eval, pred):
            rows.append(
                {
                    "Model": "RF",
                    "Candidate_ID": candidate_id,
                    "Block_ID": block["Block_ID"],
                    "Block_Length": block["Block_Length"],
                    "Gap_Category": block["Gap_Category"],
                    "Study_Pos": int(pos),
                    "Date": study_df.loc[int(pos), "Date"],
                    "Observed": float(true),
                    "Predicted": float(value),
                    "Residual": float(value - true),
                }
            )

    return pd.DataFrame(rows)


def select_and_validate_rf(
    rf_df: pd.DataFrame,
    study_df: pd.DataFrame,
    feature_columns: Sequence[str],
    blocks: Sequence[Dict[str, object]],
    candidates: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """选择最优RF参数并返回其完整块验证结果。"""
    candidate_results = []
    score_rows = []

    print("\n开始随机森林连续时间块验证……")
    for candidate_id, params in enumerate(candidates, start=1):
        result = validate_rf_candidate(
            rf_df, study_df, feature_columns, blocks, params, candidate_id
        )
        score = rf_model_score(result)
        metrics = regression_metrics(result["Observed"], result["Predicted"])
        score_rows.append(
            {
                "Candidate_ID": candidate_id,
                "Composite_Score": score,
                **metrics,
                "Parameters": json.dumps(params, ensure_ascii=False),
            }
        )
        candidate_results.append((params, result, score))
        print(
            f"  RF候选{candidate_id}：综合分数={score:.4f}，"
            f"RMSE={metrics['RMSE']:.4f}，MAE={metrics['MAE']:.4f}，R²={metrics['R2']:.4f}"
        )

    best_params, best_result, best_score = min(candidate_results, key=lambda x: x[2])
    best_result = best_result.copy()
    best_result["Model_Composite_Score"] = best_score
    print(f"RF最优参数：{best_params}")
    return best_params, best_result, pd.DataFrame(score_rows)


def fit_final_rf(
    rf_df: pd.DataFrame,
    study_df: pd.DataFrame,
    feature_columns: Sequence[str],
    best_params: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """使用研究期全部实测WT拟合最终RF，并预测研究期所有日期。"""
    study_rows = study_df["Context_Row"].to_numpy(dtype=int)
    observed_mask = study_df["WT"].notna().to_numpy()

    X_all_raw = rf_df.loc[study_rows, feature_columns].to_numpy(dtype=float)
    X_train_raw = X_all_raw[observed_mask]
    y_train = study_df.loc[observed_mask, "WT"].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_all = imputer.transform(X_all_raw)

    model = RandomForestRegressor(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)
    point_pred = model.predict(X_all)
    point_pred = np.clip(point_pred, PHYSICAL_MIN_WT, PHYSICAL_MAX_WT)

    tree_predictions = np.vstack([tree.predict(X_all) for tree in model.estimators_])
    model_std = np.std(tree_predictions, axis=0, ddof=1)

    bundle = {
        "model": model,
        "imputer": imputer,
        "feature_columns": list(feature_columns),
        "best_params": best_params,
        "study_start": str(STUDY_START.date()),
        "study_end": str(STUDY_END.date()),
    }
    return point_pred, model_std, bundle


# ============================================================
# 8. LSTM模型
# ============================================================


class WaterTemperatureLSTM(nn.Module):
    """不使用BatchNorm的稳定LSTM，避免MC Dropout与BatchNorm状态冲突。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        last = self.layer_norm(last)
        return self.head(last).squeeze(-1)


def build_sequences_for_endpoints(
    feature_matrix: np.ndarray,
    endpoint_context_rows: Sequence[int],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """按真实连续日历构建序列，绝不先删除WT缺失日期。"""
    sequences = []
    valid_rows = []
    for row in np.asarray(endpoint_context_rows, dtype=int):
        start = row - sequence_length + 1
        if start < 0:
            continue
        seq = feature_matrix[start : row + 1]
        if len(seq) != sequence_length:
            continue
        sequences.append(seq)
        valid_rows.append(row)
    if not sequences:
        return np.empty((0, sequence_length, feature_matrix.shape[1])), np.empty(0, dtype=int)
    return np.asarray(sequences, dtype=float), np.asarray(valid_rows, dtype=int)


def split_internal_validation(
    endpoint_rows: np.ndarray,
    minimum_validation: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """从训练端点中按时间顺序划出内部早停验证集。"""
    order = np.argsort(endpoint_rows)
    n = len(order)
    val_n = max(minimum_validation, int(round(n * 0.15)))
    val_n = min(val_n, max(1, n // 3))
    train_order = order[:-val_n]
    val_order = order[-val_n:]
    return train_order, val_order


def train_lstm_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: LSTMConfig,
    seed: int,
) -> Tuple[WaterTemperatureLSTM, int, List[float], List[float]]:
    """训练单个LSTM并根据内部验证集早停。"""
    set_all_seeds(seed)
    model = WaterTemperatureLSTM(
        input_size=X_train.shape[2],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.SmoothL1Loss(beta=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    batch_size = max(4, min(config.batch_size, len(train_ds)))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    best_state = None
    best_val = np.inf
    best_epoch = 1
    patience_counter = 0
    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = float(criterion(val_pred, y_val_tensor).item())

        train_history.append(train_loss)
        val_history.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("LSTM训练未产生有效模型状态。")
    model.load_state_dict(best_state)
    model.to(DEVICE)
    return model, best_epoch, train_history, val_history


def train_lstm_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: LSTMConfig,
    epochs: int,
    seed: int,
) -> WaterTemperatureLSTM:
    """用全部实测样本按固定轮数训练最终LSTM。"""
    set_all_seeds(seed)
    model = WaterTemperatureLSTM(
        input_size=X_train.shape[2],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.SmoothL1Loss(beta=0.5)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    batch_size = max(4, min(config.batch_size, len(train_ds)))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for _ in range(max(1, epochs)):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    return model


def lstm_predict(model: WaterTemperatureLSTM, X: np.ndarray) -> np.ndarray:
    """批量执行LSTM预测。"""
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        pred = model(tensor).cpu().numpy()
    return np.asarray(pred, dtype=float)


def validate_lstm_candidate(
    lstm_df: pd.DataFrame,
    study_df: pd.DataFrame,
    feature_columns: Sequence[str],
    blocks: Sequence[Dict[str, object]],
    config: LSTMConfig,
    candidate_id: int,
) -> Tuple[pd.DataFrame, List[int]]:
    """使用连续时间块验证一个LSTM结构。"""
    feature_raw = lstm_df[feature_columns].to_numpy(dtype=float)
    study_context_rows = study_df["Context_Row"].to_numpy(dtype=int)
    observed = study_df["WT"].notna().to_numpy()
    rows = []
    best_epochs: List[int] = []

    for block_index, block in enumerate(blocks):
        in_block = np.zeros(len(study_df), dtype=bool)
        in_block[int(block["Start_Pos"]) : int(block["End_Pos"]) + 1] = True
        train_study_positions = np.where(observed & (~in_block))[0]
        eval_positions = np.asarray(block["Eval_Positions"], dtype=int)

        train_endpoints = study_context_rows[train_study_positions]
        eval_endpoints = study_context_rows[eval_positions]

        X_train_raw, train_rows_valid = build_sequences_for_endpoints(
            feature_raw, train_endpoints, config.sequence_length
        )
        X_eval_raw, eval_rows_valid = build_sequences_for_endpoints(
            feature_raw, eval_endpoints, config.sequence_length
        )

        if len(X_train_raw) < 80 or len(X_eval_raw) == 0:
            raise RuntimeError(
                f"LSTM候选{candidate_id}在验证块{block['Block_ID']}上的有效序列不足。"
            )

        # 通过Context_Row映射回目标WT
        wt_by_context = lstm_df.set_index("Context_Row")["WT"]
        y_train_raw = wt_by_context.loc[train_rows_valid].to_numpy(dtype=float)
        y_eval_raw = wt_by_context.loc[eval_rows_valid].to_numpy(dtype=float)

        # 所有缩放器只在该折训练序列上拟合
        x_scaler = StandardScaler()
        x_scaler.fit(X_train_raw.reshape(-1, X_train_raw.shape[2]))
        X_train_scaled = x_scaler.transform(
            X_train_raw.reshape(-1, X_train_raw.shape[2])
        ).reshape(X_train_raw.shape)
        X_eval_scaled = x_scaler.transform(
            X_eval_raw.reshape(-1, X_eval_raw.shape[2])
        ).reshape(X_eval_raw.shape)

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()

        internal_train_idx, internal_val_idx = split_internal_validation(train_rows_valid)
        model, best_epoch, _, _ = train_lstm_with_early_stopping(
            X_train_scaled[internal_train_idx],
            y_train_scaled[internal_train_idx],
            X_train_scaled[internal_val_idx],
            y_train_scaled[internal_val_idx],
            config,
            seed=RANDOM_SEED + candidate_id * 100 + block_index,
        )
        best_epochs.append(best_epoch)

        pred_scaled = lstm_predict(model, X_eval_scaled)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        pred = np.clip(pred, PHYSICAL_MIN_WT, PHYSICAL_MAX_WT)

        # eval_rows_valid与eval_positions一一对应，因为研究期前有充足预热期
        context_to_study = dict(zip(study_context_rows, study_df["Study_Pos"]))
        valid_eval_positions = [int(context_to_study[row]) for row in eval_rows_valid]

        for pos, true, value in zip(valid_eval_positions, y_eval_raw, pred):
            rows.append(
                {
                    "Model": "LSTM",
                    "Candidate_ID": candidate_id,
                    "Block_ID": block["Block_ID"],
                    "Block_Length": block["Block_Length"],
                    "Gap_Category": block["Gap_Category"],
                    "Study_Pos": int(pos),
                    "Date": study_df.loc[int(pos), "Date"],
                    "Observed": float(true),
                    "Predicted": float(value),
                    "Residual": float(value - true),
                    "Best_Epoch": int(best_epoch),
                }
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows), best_epochs


def select_and_validate_lstm(
    lstm_df: pd.DataFrame,
    study_df: pd.DataFrame,
    feature_columns: Sequence[str],
    blocks: Sequence[Dict[str, object]],
    candidates: Sequence[LSTMConfig],
) -> Tuple[LSTMConfig, pd.DataFrame, pd.DataFrame, int]:
    """选择最优LSTM配置并返回块验证结果。"""
    candidate_results = []
    score_rows = []

    print(f"\n开始LSTM连续时间块验证，计算设备：{DEVICE}")
    for candidate_id, config in enumerate(candidates, start=1):
        result, best_epochs = validate_lstm_candidate(
            lstm_df, study_df, feature_columns, blocks, config, candidate_id
        )
        score = rf_model_score(result)
        metrics = regression_metrics(result["Observed"], result["Predicted"])
        median_epoch = int(np.median(best_epochs))
        score_rows.append(
            {
                "Candidate_ID": candidate_id,
                "Composite_Score": score,
                **metrics,
                "Median_Best_Epoch": median_epoch,
                "Configuration": json.dumps(asdict(config), ensure_ascii=False),
            }
        )
        candidate_results.append((config, result, score, median_epoch))
        print(
            f"  LSTM候选{candidate_id}：序列{config.sequence_length}天，"
            f"综合分数={score:.4f}，RMSE={metrics['RMSE']:.4f}，"
            f"MAE={metrics['MAE']:.4f}，R²={metrics['R2']:.4f}，"
            f"中位最佳轮数={median_epoch}"
        )

    best_config, best_result, best_score, best_epoch = min(
        candidate_results, key=lambda x: x[2]
    )
    best_result = best_result.copy()
    best_result["Model_Composite_Score"] = best_score
    print(f"LSTM最优配置：{best_config}")
    return best_config, best_result, pd.DataFrame(score_rows), best_epoch


def fit_final_lstm_ensemble(
    lstm_df: pd.DataFrame,
    study_df: pd.DataFrame,
    feature_columns: Sequence[str],
    config: LSTMConfig,
    epochs: int,
    seeds: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """使用研究期全部实测WT训练LSTM深度集成，并预测研究期所有日期。"""
    feature_raw = lstm_df[feature_columns].to_numpy(dtype=float)
    study_rows = study_df["Context_Row"].to_numpy(dtype=int)
    observed_positions = np.where(study_df["WT"].notna().to_numpy())[0]
    observed_endpoints = study_rows[observed_positions]

    X_train_raw, train_rows_valid = build_sequences_for_endpoints(
        feature_raw, observed_endpoints, config.sequence_length
    )
    X_all_raw, all_rows_valid = build_sequences_for_endpoints(
        feature_raw, study_rows, config.sequence_length
    )

    wt_by_context = lstm_df.set_index("Context_Row")["WT"]
    y_train_raw = wt_by_context.loc[train_rows_valid].to_numpy(dtype=float)

    x_scaler = StandardScaler()
    x_scaler.fit(X_train_raw.reshape(-1, X_train_raw.shape[2]))
    X_train = x_scaler.transform(
        X_train_raw.reshape(-1, X_train_raw.shape[2])
    ).reshape(X_train_raw.shape)
    X_all = x_scaler.transform(
        X_all_raw.reshape(-1, X_all_raw.shape[2])
    ).reshape(X_all_raw.shape)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()

    ensemble_predictions = []
    model_states = []
    final_epochs = max(20, min(int(epochs), config.max_epochs))

    print(f"\n使用全部实测水温训练LSTM深度集成，共{len(seeds)}个随机种子……")
    for seed in seeds:
        model = train_lstm_fixed_epochs(
            X_train, y_train, config, epochs=final_epochs, seed=int(seed)
        )
        pred_scaled = lstm_predict(model, X_all)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        pred = np.clip(pred, PHYSICAL_MIN_WT, PHYSICAL_MAX_WT)
        ensemble_predictions.append(pred)
        model_states.append(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = np.vstack(ensemble_predictions)
    point_pred_valid = predictions.mean(axis=0)
    std_valid = predictions.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(all_rows_valid))

    # 研究期前已有151天预热，因此所有731天均应有有效序列
    if len(all_rows_valid) != len(study_df):
        raise RuntimeError(
            f"最终LSTM只生成{len(all_rows_valid)}个预测，但研究期共有{len(study_df)}天。"
        )

    bundle = {
        "model_states": model_states,
        "config": asdict(config),
        "epochs": final_epochs,
        "seeds": list(map(int, seeds)),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_columns": list(feature_columns),
        "study_start": str(STUDY_START.date()),
        "study_end": str(STUDY_END.date()),
    }
    return point_pred_valid, std_valid, bundle


# ============================================================
# 9. 验证结果汇总、模型选择与区间校准
# ============================================================


def summarize_validation(validation_df: pd.DataFrame) -> pd.DataFrame:
    """按模型和缺口类别汇总验证指标。"""
    rows = []
    for model, model_df in validation_df.groupby("Model"):
        overall = regression_metrics(model_df["Observed"], model_df["Predicted"])
        rows.append({"Model": model, "Gap_Category": "Overall", **overall})
        for category, cat_df in model_df.groupby("Gap_Category"):
            rows.append(
                {
                    "Model": model,
                    "Gap_Category": category,
                    **regression_metrics(cat_df["Observed"], cat_df["Predicted"]),
                }
            )
    return pd.DataFrame(rows)


def compare_machine_learning_models(validation_df: pd.DataFrame) -> pd.DataFrame:
    """仅在RF和LSTM之间计算最终模型选择分数。"""
    rows = []
    for model in ["RF", "LSTM"]:
        model_df = validation_df[validation_df["Model"] == model]
        if model_df.empty:
            continue
        overall = regression_metrics(model_df["Observed"], model_df["Predicted"])
        long_df = model_df[model_df["Gap_Category"].isin(["16-60 days", ">60 days"])]
        long_metrics = regression_metrics(long_df["Observed"], long_df["Predicted"])
        long_rmse = long_metrics["RMSE"] if np.isfinite(long_metrics["RMSE"]) else overall["RMSE"]
        score = (
            0.45 * overall["RMSE"]
            + 0.20 * overall["MAE"]
            + 0.25 * long_rmse
            + 0.10 * abs(overall["Bias"])
        )
        rows.append(
            {
                "Model": model,
                "Selection_Score": float(score),
                "Overall_RMSE": overall["RMSE"],
                "Overall_MAE": overall["MAE"],
                "Overall_Bias": overall["Bias"],
                "Overall_R2": overall["R2"],
                "Long_Gap_RMSE": long_rmse,
                "Validation_N": overall["N"],
            }
        )
    result = pd.DataFrame(rows).sort_values("Selection_Score").reset_index(drop=True)
    if result.empty:
        raise RuntimeError("RF和LSTM均无有效验证结果，无法选择最终模型。")
    return result


def calibrate_conformal_intervals(
    selected_validation: pd.DataFrame,
    confidence: float = 0.95,
    minimum_category_samples: int = 15,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """按照模拟缺口长度类别校准保形预测区间。"""
    residual_abs = np.abs(
        selected_validation["Predicted"].to_numpy(dtype=float)
        - selected_validation["Observed"].to_numpy(dtype=float)
    )
    global_q = finite_sample_quantile(residual_abs, confidence=confidence)
    q_by_category: Dict[str, float] = {}
    rows = []

    for category in ["1-3 days", "4-15 days", "16-60 days", ">60 days"]:
        cat_df = selected_validation[selected_validation["Gap_Category"] == category]
        cat_abs = np.abs(cat_df["Predicted"] - cat_df["Observed"]).to_numpy(dtype=float)
        if len(cat_abs) >= minimum_category_samples:
            q = finite_sample_quantile(cat_abs, confidence=confidence)
            source = "Category-specific"
        else:
            q = global_q
            source = "Global fallback"
        q_by_category[category] = float(q)

        if len(cat_df) > 0:
            lower = cat_df["Predicted"] - q
            upper = cat_df["Predicted"] + q
            coverage = np.mean(
                (cat_df["Observed"] >= lower) & (cat_df["Observed"] <= upper)
            )
        else:
            coverage = np.nan

        rows.append(
            {
                "Gap_Category": category,
                "Calibration_N": int(len(cat_abs)),
                "Half_Width_95": float(q),
                "Calibration_Source": source,
                "Empirical_Coverage": float(coverage) if np.isfinite(coverage) else np.nan,
            }
        )

    lower_global = selected_validation["Predicted"] - global_q
    upper_global = selected_validation["Predicted"] + global_q
    global_coverage = np.mean(
        (selected_validation["Observed"] >= lower_global)
        & (selected_validation["Observed"] <= upper_global)
    )
    rows.append(
        {
            "Gap_Category": "Overall",
            "Calibration_N": int(len(selected_validation)),
            "Half_Width_95": float(global_q),
            "Calibration_Source": "Global",
            "Empirical_Coverage": float(global_coverage),
        }
    )
    return q_by_category, pd.DataFrame(rows)


def evaluate_intervals_on_independent_test(
    selected_test: pd.DataFrame,
    q_by_category: Dict[str, float],
) -> pd.DataFrame:
    """在独立测试时间块上评价已经校准好的95%预测区间。"""
    rows = []
    for category in ["1-3 days", "4-15 days", "16-60 days", ">60 days"]:
        cat_df = selected_test[selected_test["Gap_Category"] == category].copy()
        if cat_df.empty:
            rows.append({
                "Gap_Category": category,
                "Test_N": 0,
                "Half_Width_95": q_by_category.get(category, np.nan),
                "Independent_Coverage": np.nan,
                "Mean_Interval_Width": np.nan,
                "Winkler_Score": np.nan,
            })
            continue
        q = float(q_by_category[category])
        lower = cat_df["Predicted"].to_numpy(dtype=float) - q
        upper = cat_df["Predicted"].to_numpy(dtype=float) + q
        observed = cat_df["Observed"].to_numpy(dtype=float)
        coverage = np.mean((observed >= lower) & (observed <= upper))
        alpha = 0.05
        width = upper - lower
        penalty_low = (2.0 / alpha) * (lower - observed) * (observed < lower)
        penalty_high = (2.0 / alpha) * (observed - upper) * (observed > upper)
        winkler = np.mean(width + penalty_low + penalty_high)
        rows.append({
            "Gap_Category": category,
            "Test_N": int(len(cat_df)),
            "Half_Width_95": q,
            "Independent_Coverage": float(coverage),
            "Mean_Interval_Width": float(np.mean(width)),
            "Winkler_Score": float(winkler),
        })

    if len(selected_test) > 0:
        q_values = selected_test["Gap_Category"].map(q_by_category).to_numpy(dtype=float)
        lower = selected_test["Predicted"].to_numpy(dtype=float) - q_values
        upper = selected_test["Predicted"].to_numpy(dtype=float) + q_values
        observed = selected_test["Observed"].to_numpy(dtype=float)
        coverage = np.mean((observed >= lower) & (observed <= upper))
        alpha = 0.05
        width = upper - lower
        penalty_low = (2.0 / alpha) * (lower - observed) * (observed < lower)
        penalty_high = (2.0 / alpha) * (observed - upper) * (observed > upper)
        winkler = np.mean(width + penalty_low + penalty_high)
        rows.append({
            "Gap_Category": "Overall",
            "Test_N": int(len(selected_test)),
            "Half_Width_95": float(np.mean(q_values)),
            "Independent_Coverage": float(coverage),
            "Mean_Interval_Width": float(np.mean(width)),
            "Winkler_Score": float(winkler),
        })
    return pd.DataFrame(rows)


def estimate_residual_ar1(selected_validation: pd.DataFrame) -> float:
    """从各连续验证块残差估计多重插补使用的一阶时间相关系数。"""
    pair_x = []
    pair_y = []
    for _, block_df in selected_validation.groupby("Block_ID"):
        block_df = block_df.sort_values("Date")
        residual = block_df["Residual"].to_numpy(dtype=float)
        positions = block_df["Study_Pos"].to_numpy(dtype=int)
        for i in range(1, len(residual)):
            if positions[i] - positions[i - 1] == 1:
                pair_x.append(residual[i - 1])
                pair_y.append(residual[i])
    if len(pair_x) < 8 or np.std(pair_x) == 0 or np.std(pair_y) == 0:
        return 0.5
    rho = float(np.corrcoef(pair_x, pair_y)[0, 1])
    if not np.isfinite(rho):
        return 0.5
    return float(np.clip(rho, 0.0, 0.90))


# ============================================================
# 10. 最终结果、多重插补和输出
# ============================================================


def build_final_result(
    study_df_with_gaps: pd.DataFrame,
    point_prediction: np.ndarray,
    model_std: np.ndarray,
    selected_model: str,
    q_by_category: Dict[str, float],
    maximum_validated_gap: int,
) -> pd.DataFrame:
    """构建最终研究期731天填补结果。"""
    result = study_df_with_gaps.copy()
    result["WT_Observed"] = result["WT"]
    result["WT_Model_Prediction"] = np.asarray(point_prediction, dtype=float)
    result["WT_Model_SD"] = np.asarray(model_std, dtype=float)
    result["Is_Imputed"] = result["WT_Observed"].isna()
    result["Selected_Model"] = selected_model

    result["WT_Final"] = result["WT_Observed"]
    result.loc[result["Is_Imputed"], "WT_Final"] = result.loc[
        result["Is_Imputed"], "WT_Model_Prediction"
    ]
    result["WT_Final"] = result["WT_Final"].clip(PHYSICAL_MIN_WT, PHYSICAL_MAX_WT)

    result["Prediction_Half_Width_95"] = np.nan
    result["Prediction_Lower_95"] = np.nan
    result["Prediction_Upper_95"] = np.nan
    result["Prediction_SD_Approx"] = np.nan

    for idx in result.index[result["Is_Imputed"]]:
        category = result.at[idx, "Gap_Category"]
        half_width = float(q_by_category.get(category, np.nan))
        pred = float(result.at[idx, "WT_Final"])
        result.at[idx, "Prediction_Half_Width_95"] = half_width
        result.at[idx, "Prediction_Lower_95"] = max(
            PHYSICAL_MIN_WT, pred - half_width
        )
        result.at[idx, "Prediction_Upper_95"] = min(
            PHYSICAL_MAX_WT, pred + half_width
        )
        result.at[idx, "Prediction_SD_Approx"] = half_width / 1.96

    # 超出验证长度或位于研究期边界的缺失段标记为外推风险
    result["Extrapolation_Flag"] = False
    missing_groups = result[result["Is_Imputed"]].groupby("Gap_ID")
    for gap_id, group in missing_groups:
        length = int(group["Gap_Length"].iloc[0])
        starts_at_boundary = int(group.index.min()) == 0
        ends_at_boundary = int(group.index.max()) == len(result) - 1
        flag = length > maximum_validated_gap or starts_at_boundary or ends_at_boundary
        result.loc[group.index, "Extrapolation_Flag"] = bool(flag)

    result["Date"] = pd.to_datetime(result["Date"])
    return result


def generate_multiple_imputations(
    final_result: pd.DataFrame,
    q_by_category: Dict[str, float],
    rho: float,
    n_imputations: int,
    seed: int,
) -> pd.DataFrame:
    """
    生成具有缺口内时间相关性的多重插补序列。

    观测值在所有插补序列中保持不变；缺失值围绕最终模型点预测波动。
    """
    rng = np.random.default_rng(seed)
    base = final_result["WT_Final"].to_numpy(dtype=float)
    observed = final_result["WT_Observed"].notna().to_numpy()
    output = pd.DataFrame(
        {
            "Date": final_result["Date"].dt.strftime("%Y-%m-%d"),
            "JDY": final_result["JDY"].astype(int),
            "WT_Observed": final_result["WT_Observed"],
        }
    )

    gap_groups = [group.index.to_numpy(dtype=int) for _, group in final_result[final_result["Is_Imputed"]].groupby("Gap_ID")]

    for m in range(1, n_imputations + 1):
        values = base.copy()
        for indices in gap_groups:
            category = str(final_result.loc[indices[0], "Gap_Category"])
            half_width = float(q_by_category[category])
            sigma = max(half_width / 1.96, 1e-6)
            residual = np.zeros(len(indices), dtype=float)
            residual[0] = rng.normal(0.0, sigma)
            innovation_sd = sigma * math.sqrt(max(1e-8, 1.0 - rho ** 2))
            for i in range(1, len(indices)):
                residual[i] = rho * residual[i - 1] + rng.normal(0.0, innovation_sd)
            values[indices] = np.clip(
                base[indices] + residual,
                PHYSICAL_MIN_WT,
                PHYSICAL_MAX_WT,
            )
        values[observed] = final_result.loc[observed, "WT_Observed"].to_numpy(dtype=float)
        output[f"WT_MI_{m:03d}"] = values

    return output


def save_model_files(
    output_dir: Path,
    rf_bundle: Dict[str, object],
    lstm_bundle: Optional[Dict[str, object]],
    selected_model: str,
) -> None:
    """保存RF和LSTM模型文件及模型选择信息。"""
    joblib.dump(rf_bundle, output_dir / "RF_final_model_bundle.joblib")
    if lstm_bundle is not None:
        torch.save(lstm_bundle, output_dir / "LSTM_final_ensemble_bundle.pth")
    with open(output_dir / "最终模型选择.txt", "w", encoding="utf-8") as f:
        f.write(f"最终选择模型：{selected_model}\n")
        f.write(f"正式研究期：{STUDY_START.date()} 至 {STUDY_END.date()}\n")
        f.write("2023-01-01至2023-05-31仅用于预测因子特征预热。\n")


def save_tables(
    output_dir: Path,
    blocks_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_summary_df: pd.DataFrame,
    calibration_model_comparison: pd.DataFrame,
    independent_test_comparison: pd.DataFrame,
    interval_calibration: pd.DataFrame,
    interval_test_evaluation: pd.DataFrame,
    rf_candidate_scores: pd.DataFrame,
    lstm_candidate_scores: Optional[pd.DataFrame],
    final_result: pd.DataFrame,
    multiple_imputations: pd.DataFrame,
    n_multiple_imputations: int,
) -> None:
    """保存CSV和Excel汇总文件。"""
    blocks_df.to_csv(output_dir / "01_连续时间块验证设计.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(output_dir / "02_调参校准测试逐点结果.csv", index=False, encoding="utf-8-sig")
    test_summary_df.to_csv(output_dir / "03_独立测试性能分缺口类别汇总.csv", index=False, encoding="utf-8-sig")
    calibration_model_comparison.to_csv(output_dir / "04_RF_LSTM校准集模型选择.csv", index=False, encoding="utf-8-sig")
    independent_test_comparison.to_csv(output_dir / "05_RF_LSTM独立测试性能.csv", index=False, encoding="utf-8-sig")
    interval_calibration.to_csv(output_dir / "06_95预测区间校准结果.csv", index=False, encoding="utf-8-sig")
    interval_test_evaluation.to_csv(output_dir / "07_95预测区间独立测试结果.csv", index=False, encoding="utf-8-sig")
    rf_candidate_scores.to_csv(output_dir / "08_RF候选参数调参结果.csv", index=False, encoding="utf-8-sig")
    if lstm_candidate_scores is not None:
        lstm_candidate_scores.to_csv(output_dir / "09_LSTM候选结构调参结果.csv", index=False, encoding="utf-8-sig")

    output_columns = [
        "Date", "JDY", "WT_Observed", "WT_Final", "Is_Imputed",
        "Gap_ID", "Gap_Length", "Gap_Category", "Selected_Model",
        "WT_Model_Prediction", "WT_Model_SD", "Prediction_Lower_95",
        "Prediction_Upper_95", "Prediction_Half_Width_95",
        "Prediction_SD_Approx", "Extrapolation_Flag",
        "QIN", "TAIR", "TDEW", "WS", "WD", "CLOUD",
    ]
    final_to_save = final_result[output_columns].copy()
    final_to_save["Date"] = final_to_save["Date"].dt.strftime("%Y-%m-%d")
    final_to_save.to_csv(
        output_dir / "10_研究期水温最终填补结果.csv", index=False, encoding="utf-8-sig"
    )
    multiple_imputations.to_csv(
        output_dir / f"11_水温多重插补{n_multiple_imputations}组.csv",
        index=False,
        encoding="utf-8-sig",
    )

    try:
        with pd.ExcelWriter(output_dir / "水温填补综合结果.xlsx", engine="openpyxl") as writer:
            blocks_df.to_excel(writer, sheet_name="验证块设计", index=False)
            validation_df.to_excel(writer, sheet_name="逐点验证", index=False)
            test_summary_df.to_excel(writer, sheet_name="独立测试汇总", index=False)
            calibration_model_comparison.to_excel(writer, sheet_name="校准集模型选择", index=False)
            independent_test_comparison.to_excel(writer, sheet_name="独立测试性能", index=False)
            interval_calibration.to_excel(writer, sheet_name="区间校准", index=False)
            interval_test_evaluation.to_excel(writer, sheet_name="区间独立测试", index=False)
            rf_candidate_scores.to_excel(writer, sheet_name="RF调参", index=False)
            if lstm_candidate_scores is not None:
                lstm_candidate_scores.to_excel(writer, sheet_name="LSTM调参", index=False)
            final_to_save.to_excel(writer, sheet_name="最终填补结果", index=False)
    except Exception as exc:
        print(f"警告：综合Excel保存失败，但CSV结果已正常保存。原因：{exc}")


# ============================================================
# 11. 图件输出
# ============================================================


def plot_model_comparison(summary_df: pd.DataFrame, figure_dir: Path) -> None:
    """图1：不同缺口类别下各模型RMSE。"""
    order = ["Overall", "1-3 days", "4-15 days", "16-60 days", ">60 days"]
    models = [m for m in ["Linear", "PCHIP", "RF", "LSTM"] if m in summary_df["Model"].unique()]

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    x = np.arange(len(order))
    width = 0.18 if len(models) >= 4 else 0.24

    for i, model in enumerate(models):
        values = []
        for category in order:
            row = summary_df[(summary_df["Model"] == model) & (summary_df["Gap_Category"] == category)]
            values.append(float(row["RMSE"].iloc[0]) if not row.empty else np.nan)
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=model, edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Validation gap category", fontsize=11)
    ax.set_ylabel("RMSE (°C)", fontsize=11)
    ax.set_title("Model performance under continuous-block validation", fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.legend(frameon=False, ncol=min(4, len(models)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Figure1_model_comparison")


def plot_oof_scatter(selected_validation: pd.DataFrame, selected_model: str, figure_dir: Path) -> None:
    """图2：最终模型连续块验证的观测值与预测值。"""
    metrics = regression_metrics(selected_validation["Observed"], selected_validation["Predicted"])
    fig, ax = plt.subplots(figsize=(6.8, 6.3))
    scatter = ax.scatter(
        selected_validation["Observed"],
        selected_validation["Predicted"],
        c=selected_validation["Block_Length"],
        cmap="viridis",
        s=38,
        alpha=0.78,
        edgecolors="black",
        linewidth=0.35,
    )
    low = min(selected_validation["Observed"].min(), selected_validation["Predicted"].min())
    high = max(selected_validation["Observed"].max(), selected_validation["Predicted"].max())
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("Observed water temperature (°C)", fontsize=11)
    ax.set_ylabel("Predicted water temperature (°C)", fontsize=11)
    ax.set_title(f"{selected_model} out-of-block validation", fontsize=12, pad=10)
    text = (
        f"RMSE = {metrics['RMSE']:.2f} °C\n"
        f"MAE = {metrics['MAE']:.2f} °C\n"
        f"R² = {metrics['R2']:.2f}\n"
        f"n = {metrics['N']}"
    )
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", fontsize=10)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Masked block length (days)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Figure2_selected_model_oof_scatter")


def plot_final_time_series(final_result: pd.DataFrame, figure_dir: Path) -> None:
    """图3：研究期实测、填补值和95%预测区间。"""
    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    observed_mask = ~final_result["Is_Imputed"]
    missing_mask = final_result["Is_Imputed"]

    ax.plot(
        final_result["Date"],
        final_result["WT_Final"],
        linewidth=1.0,
        color="0.55",
        label="Completed series",
        zorder=1,
    )
    ax.scatter(
        final_result.loc[observed_mask, "Date"],
        final_result.loc[observed_mask, "WT_Observed"],
        s=12,
        color="black",
        label="Observed",
        zorder=3,
    )
    ax.plot(
        final_result.loc[missing_mask, "Date"],
        final_result.loc[missing_mask, "WT_Final"],
        linewidth=1.2,
        color="#C44E52",
        label="Imputed",
        zorder=4,
    )
    ax.fill_between(
        final_result.loc[missing_mask, "Date"],
        final_result.loc[missing_mask, "Prediction_Lower_95"],
        final_result.loc[missing_mask, "Prediction_Upper_95"],
        color="#C44E52",
        alpha=0.22,
        label="95% prediction interval",
        zorder=2,
    )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Water temperature (°C)", fontsize=11)
    ax.set_title("Observed and imputed water temperature during the two hydrological years", fontsize=12, pad=10)
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(frameon=False, ncol=4, loc="upper center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Figure3_final_imputed_time_series")


def plot_residual_by_gap(selected_validation: pd.DataFrame, figure_dir: Path) -> None:
    """图4：不同模拟缺口类别下残差分布。"""
    order = ["1-3 days", "4-15 days", "16-60 days", ">60 days"]
    data = [
        selected_validation.loc[selected_validation["Gap_Category"] == category, "Residual"].to_numpy(dtype=float)
        for category in order
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.boxplot(data, labels=order, showfliers=True, widths=0.58)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
    ax.set_xlabel("Validation gap category", fontsize=11)
    ax.set_ylabel("Prediction residual (°C)", fontsize=11)
    ax.set_title("Residual distribution across simulated gap lengths", fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Figure4_residual_by_gap_category")


# ============================================================
# 12. 中文说明报告
# ============================================================


def write_chinese_report(
    output_dir: Path,
    study_df: pd.DataFrame,
    blocks: Sequence[Dict[str, object]],
    calibration_model_comparison: pd.DataFrame,
    independent_test_comparison: pd.DataFrame,
    interval_calibration: pd.DataFrame,
    interval_test_evaluation: pd.DataFrame,
    selected_model: str,
    best_rf_params: Dict[str, object],
    best_lstm_config: Optional[LSTMConfig],
    residual_rho: float,
    final_result: pd.DataFrame,
    n_multiple_imputations: int,
) -> None:
    """生成中文模型方法与结果说明。"""
    report_path = output_dir / "水温填补模型说明报告.txt"
    selected_cal_row = calibration_model_comparison[
        calibration_model_comparison["Model"] == selected_model
    ].iloc[0]
    selected_test_row = independent_test_comparison[
        independent_test_comparison["Model"] == selected_model
    ].iloc[0]
    long_gaps = final_result[final_result["Is_Imputed"]].groupby("Gap_ID")["Gap_Length"].first()

    role_counts = pd.Series([b.get("Validation_Role", "未标记") for b in blocks]).value_counts().to_dict()

    lines = [
        "水温缺失值填补模型说明报告",
        "=" * 90,
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "一、时间范围使用规则",
        "-" * 90,
        f"1. 论文正式研究期为{STUDY_START.date()}至{STUDY_END.date()}，共{len(study_df)}天。",
        f"2. 研究期内水温实测{study_df['WT'].notna().sum()}天，缺失{study_df['WT'].isna().sum()}天，缺失率{study_df['WT'].isna().mean()*100:.2f}%。",
        f"3. {WARMUP_START.date()}至{(STUDY_START-pd.Timedelta(days=1)).date()}只用于生成预测因子的滞后、滑动统计和LSTM历史序列，不使用该阶段WT作为正式训练目标，也不计入论文样本量。",
        f"4. {STUDY_END.date()}之后的数据未参与训练、验证、模型选择或正式填补结果。",
        "",
        "二、连续时间块验证设计",
        "-" * 90,
        f"程序共构建{len(blocks)}个连续时间块，验证用途数量为：{role_counts}。",
        f"模拟缺口长度包括：{VALIDATION_BLOCK_LENGTHS}天。",
        "Tuning时间块只用于RF参数和LSTM结构选择；Calibration时间块用于RF与LSTM最终模型选择及95%预测区间校准；Test时间块只用于独立性能报告。",
        "每次验证均隐藏一个完整时间块内的实测水温，再使用时间块外的水温训练模型，最后只在时间块内原本有实测值的日期评价误差。",
        "线性插值、PCHIP、RF和LSTM在相同Test时间块上比较。",
        "",
        "三、泄漏控制和旧代码修正",
        "-" * 90,
        "1. 取消随机8:2拆分，改为连续时间块验证。",
        "2. 调参、模型选择、区间校准和独立测试使用不同时间块，避免测试集被重复用于选模型。",
        "3. 不再使用center=True的居中移动平均，所有RF移动特征只包含当日及过去信息。",
        "4. 每个验证折的填补器、标准化器和模型只在该折训练样本上拟合。",
        "5. LSTM序列基于完整连续日历构建，不先删除WT缺失行，因此不会把不相邻日期错误拼接成连续序列。",
        "6. WD原始单位为弧度，直接计算sin(WD)和cos(WD)。",
        "7. 移除旧RF代码中人为加入的随机噪声、无效二次插值和结果后平滑。",
        "8. LSTM移除BatchNorm与MC Dropout冲突，最终采用多随机种子深度集成，并使用保形方法校准预测区间。",
        "",
        "四、Calibration时间块上的RF与LSTM模型选择",
        "-" * 90,
        calibration_model_comparison.to_string(index=False),
        "",
        f"最终选择模型：{selected_model}",
        f"校准集模型选择综合分数：{selected_cal_row['Selection_Score']:.4f}",
        "",
        "五、独立Test时间块性能",
        "-" * 90,
        independent_test_comparison.to_string(index=False),
        "",
        f"最终模型独立测试RMSE：{selected_test_row['Overall_RMSE']:.4f} °C",
        f"最终模型独立测试MAE：{selected_test_row['Overall_MAE']:.4f} °C",
        f"最终模型独立测试Bias：{selected_test_row['Overall_Bias']:.4f} °C",
        f"最终模型独立测试R²：{selected_test_row['Overall_R2']:.4f}",
        f"最终模型长缺口独立测试RMSE：{selected_test_row['Long_Gap_RMSE']:.4f} °C",
        "",
        "六、模型参数",
        "-" * 90,
        f"RF最优参数：{json.dumps(best_rf_params, ensure_ascii=False)}",
        f"LSTM最优配置：{asdict(best_lstm_config) if best_lstm_config is not None else '未运行LSTM'}",
        "",
        "七、95%预测区间",
        "-" * 90,
        "预测区间根据Calibration时间块残差按缺口长度类别进行保形校准，不再使用所有日期统一的正态残差标准差。",
        "Calibration结果：",
        interval_calibration.to_string(index=False),
        "",
        "独立Test时间块覆盖率：",
        interval_test_evaluation.to_string(index=False),
        f"多重插补缺口内残差的一阶相关系数估计值：{residual_rho:.3f}",
        "",
        "八、最终填补结果",
        "-" * 90,
        f"共填补{final_result['Is_Imputed'].sum()}个研究期水温缺失值。",
        f"真实缺失段数量：{final_result.loc[final_result['Is_Imputed'], 'Gap_ID'].nunique()}。",
        f"最长真实连续缺失：{int(long_gaps.max()) if len(long_gaps)>0 else 0}天。",
        f"被标记为外推风险的填补天数：{int(final_result['Extrapolation_Flag'].sum())}天。",
        f"同时生成{n_multiple_imputations}组具有缺口内时间相关性的多重插补水温序列，供后续氮通量和机理模型进行不确定性传播。",
        "",
        "九、论文中应明确报告的内容",
        "-" * 90,
        "1. 分别说明正式研究期、特征预热期和研究期外数据的用途。",
        "2. 报告研究期内原始实测数量、缺失率、连续缺口数量和最长缺口。",
        "3. 报告Tuning、Calibration和Test连续时间块的设置，而不是随机8:2拆分。",
        "4. 报告RF和LSTM关键参数、模型选择规则、独立测试性能及分缺口长度性能。",
        "5. 将Prediction_Lower_95和Prediction_Upper_95称为95%预测区间，不称为均值置信区间。",
        "6. 后续氮通量或CE-QUAL-W2模拟应分别使用多组插补序列，以传播输入填补不确定性。",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 13. 主程序
# ============================================================


def main() -> None:
    """执行完整的RF与LSTM水温填补、比较、区间校准和输出流程。"""
    output_dir = ensure_dir(OUTPUT_DIR)
    figure_dir = ensure_dir(output_dir / "figures")

    print("=" * 90)
    print("RF + LSTM 水温缺失值填补统一程序（审稿修订版）")
    print(f"PyTorch设备：{DEVICE}；CPU线程数：{TORCH_NUM_THREADS}")
    print(f"快速测试模式：{QUICK_MODE}")
    print("=" * 90)

    context_df, study_df = load_and_prepare_data(INPUT_FILE)
    study_df = assign_actual_gap_information(study_df)

    rf_df, rf_features = create_rf_features(context_df)
    lstm_df, lstm_features = create_lstm_features(context_df)

    if QUICK_MODE:
        block_lengths = [7, 30, 90]
        blocks_per_length = 1
        rf_candidates = RF_CANDIDATES[:1]
        lstm_candidates = [LSTMConfig(
            sequence_length=14, hidden_size=24, num_layers=1, dropout=0.15,
            max_epochs=35, patience=6,
        )]
        lstm_seeds = [42]
        n_multiple = 10
    else:
        block_lengths = VALIDATION_BLOCK_LENGTHS
        blocks_per_length = BLOCKS_PER_LENGTH
        rf_candidates = RF_CANDIDATES
        lstm_candidates = LSTM_CANDIDATES
        lstm_seeds = LSTM_ENSEMBLE_SEEDS
        n_multiple = N_MULTIPLE_IMPUTATIONS

    blocks = generate_validation_blocks(
        study_df,
        block_lengths=block_lengths,
        blocks_per_length=blocks_per_length,
        random_seed=RANDOM_SEED,
    )
    blocks_df = blocks_to_dataframe(blocks)

    if QUICK_MODE:
        # 快速模式仅检查程序链路，不能作为论文正式性能结果。
        tuning_blocks = blocks
        calibration_blocks = blocks
        test_blocks = blocks
        print("警告：快速测试模式重复使用相同时间块，不可用于论文正式报告。")
    else:
        tuning_blocks = [b for b in blocks if b.get("Validation_Role") == "Tuning"]
        calibration_blocks = [b for b in blocks if b.get("Validation_Role") == "Calibration"]
        test_blocks = [b for b in blocks if b.get("Validation_Role") == "Test"]
        if not tuning_blocks or not calibration_blocks or not test_blocks:
            raise RuntimeError(
                "正式模式必须同时具有Tuning、Calibration和Test时间块；"
                "请将BLOCKS_PER_LENGTH设置为3或更大。"
            )

    # --------------------------------------------------------
    # A. 调参：只使用Tuning时间块
    # --------------------------------------------------------
    best_rf_params, rf_tuning, rf_candidate_scores = select_and_validate_rf(
        rf_df, study_df, rf_features, tuning_blocks, candidates=rf_candidates
    )
    rf_tuning = rf_tuning.copy()
    rf_tuning["Validation_Role"] = "Tuning" if not QUICK_MODE else "QuickCheck"

    rf_calibration = validate_rf_candidate(
        rf_df, study_df, rf_features, calibration_blocks, best_rf_params, candidate_id=900
    )
    rf_calibration["Validation_Role"] = "Calibration" if not QUICK_MODE else "QuickCheck"

    rf_test = validate_rf_candidate(
        rf_df, study_df, rf_features, test_blocks, best_rf_params, candidate_id=901
    )
    rf_test["Validation_Role"] = "Test" if not QUICK_MODE else "QuickCheck"

    rf_point, rf_std, rf_bundle = fit_final_rf(
        rf_df, study_df, rf_features, best_rf_params
    )

    # --------------------------------------------------------
    # B. LSTM调参与独立验证
    # --------------------------------------------------------
    lstm_tuning = pd.DataFrame()
    lstm_calibration = pd.DataFrame()
    lstm_test = pd.DataFrame()
    lstm_candidate_scores: Optional[pd.DataFrame] = None
    best_lstm_config: Optional[LSTMConfig] = None
    lstm_bundle: Optional[Dict[str, object]] = None
    lstm_point: Optional[np.ndarray] = None
    lstm_std: Optional[np.ndarray] = None

    if RUN_LSTM:
        best_lstm_config, lstm_tuning, lstm_candidate_scores, best_lstm_epoch = select_and_validate_lstm(
            lstm_df, study_df, lstm_features, tuning_blocks, candidates=lstm_candidates
        )
        lstm_tuning = lstm_tuning.copy()
        lstm_tuning["Validation_Role"] = "Tuning" if not QUICK_MODE else "QuickCheck"

        lstm_calibration, calibration_epochs = validate_lstm_candidate(
            lstm_df, study_df, lstm_features, calibration_blocks,
            best_lstm_config, candidate_id=900
        )
        lstm_calibration["Validation_Role"] = "Calibration" if not QUICK_MODE else "QuickCheck"

        lstm_test, test_epochs = validate_lstm_candidate(
            lstm_df, study_df, lstm_features, test_blocks,
            best_lstm_config, candidate_id=901
        )
        lstm_test["Validation_Role"] = "Test" if not QUICK_MODE else "QuickCheck"

        # 最终训练轮数综合调参、校准和测试各折早停结果的中位数，
        # 但最终模型使用研究期全部实测水温重新训练。
        epoch_pool = [best_lstm_epoch] + calibration_epochs
        final_lstm_epochs = int(np.median(epoch_pool))
        lstm_point, lstm_std, lstm_bundle = fit_final_lstm_ensemble(
            lstm_df, study_df, lstm_features, best_lstm_config,
            epochs=final_lstm_epochs, seeds=lstm_seeds
        )

    # --------------------------------------------------------
    # C. Calibration时间块：选择RF或LSTM并校准区间
    # --------------------------------------------------------
    calibration_parts = [rf_calibration]
    if not lstm_calibration.empty:
        calibration_parts.append(lstm_calibration)
    calibration_ml = pd.concat(calibration_parts, ignore_index=True)
    calibration_model_comparison = compare_machine_learning_models(calibration_ml)
    selected_model = str(calibration_model_comparison.iloc[0]["Model"])

    print("\nCalibration时间块上的RF与LSTM比较：")
    print(calibration_model_comparison.to_string(index=False))
    print(f"\n最终选择模型：{selected_model}")

    selected_calibration = calibration_ml[calibration_ml["Model"] == selected_model].copy()
    q_by_category, interval_calibration = calibrate_conformal_intervals(
        selected_calibration, confidence=0.95, minimum_category_samples=15
    )
    residual_rho = estimate_residual_ar1(selected_calibration)

    # --------------------------------------------------------
    # D. Test时间块：独立性能和区间覆盖率评价
    # --------------------------------------------------------
    test_ml_parts = [rf_test]
    if not lstm_test.empty:
        test_ml_parts.append(lstm_test)
    test_ml = pd.concat(test_ml_parts, ignore_index=True)
    independent_test_comparison = compare_machine_learning_models(test_ml)
    selected_test = test_ml[test_ml["Model"] == selected_model].copy()
    interval_test_evaluation = evaluate_intervals_on_independent_test(
        selected_test, q_by_category
    )

    print("\n独立Test时间块性能（不参与模型选择）：")
    print(independent_test_comparison.to_string(index=False))
    print("\n95%预测区间独立测试结果：")
    print(interval_test_evaluation.to_string(index=False))

    baseline_test = validate_interpolation_baselines(study_df, test_blocks)
    baseline_test["Validation_Role"] = "Test" if not QUICK_MODE else "QuickCheck"
    test_validation = pd.concat([baseline_test, test_ml], ignore_index=True)
    test_summary_df = summarize_validation(test_validation)

    # 汇总所有逐点结果
    validation_parts = [rf_tuning, rf_calibration, rf_test]
    if not lstm_tuning.empty:
        validation_parts.extend([lstm_tuning, lstm_calibration, lstm_test])
    validation_parts.append(baseline_test)
    validation_df = pd.concat(validation_parts, ignore_index=True, sort=False)

    # --------------------------------------------------------
    # E. 使用全部实测WT拟合最终模型并填补299天
    # --------------------------------------------------------
    if selected_model == "RF":
        selected_point = rf_point
        selected_std = rf_std
    elif selected_model == "LSTM" and lstm_point is not None and lstm_std is not None:
        selected_point = lstm_point
        selected_std = lstm_std
    else:
        raise RuntimeError("最终模型选择结果与可用预测不一致。")

    final_result = build_final_result(
        study_df, selected_point, selected_std, selected_model, q_by_category,
        maximum_validated_gap=max(block_lengths),
    )
    multiple_imputations = generate_multiple_imputations(
        final_result, q_by_category=q_by_category, rho=residual_rho,
        n_imputations=n_multiple, seed=RANDOM_SEED + 999,
    )

    save_tables(
        output_dir, blocks_df, validation_df, test_summary_df,
        calibration_model_comparison, independent_test_comparison,
        interval_calibration, interval_test_evaluation,
        rf_candidate_scores, lstm_candidate_scores, final_result,
        multiple_imputations, n_multiple_imputations=n_multiple,
    )
    save_model_files(output_dir, rf_bundle, lstm_bundle, selected_model)

    plot_model_comparison(test_summary_df, figure_dir)
    plot_oof_scatter(selected_test, selected_model, figure_dir)
    plot_final_time_series(final_result, figure_dir)
    plot_residual_by_gap(selected_test, figure_dir)

    write_chinese_report(
        output_dir, study_df, blocks, calibration_model_comparison,
        independent_test_comparison, interval_calibration,
        interval_test_evaluation, selected_model, best_rf_params,
        best_lstm_config, residual_rho, final_result, n_multiple,
    )

    run_config = {
        "input_file": INPUT_FILE,
        "output_dir": OUTPUT_DIR,
        "base_date": str(BASE_DATE.date()),
        "warmup_start": str(WARMUP_START.date()),
        "study_start": str(STUDY_START.date()),
        "study_end": str(STUDY_END.date()),
        "validation_block_lengths": block_lengths,
        "blocks_per_length": blocks_per_length,
        "random_seed": RANDOM_SEED,
        "quick_mode": QUICK_MODE,
        "run_lstm": RUN_LSTM,
        "selected_model": selected_model,
        "n_multiple_imputations": n_multiple,
        "rf_best_params": best_rf_params,
        "lstm_best_config": asdict(best_lstm_config) if best_lstm_config else None,
        "conformal_half_widths": q_by_category,
        "residual_ar1": residual_rho,
        "validation_roles": {
            "tuning_blocks": len(tuning_blocks),
            "calibration_blocks": len(calibration_blocks),
            "test_blocks": len(test_blocks),
        },
    }
    with open(output_dir / "运行参数.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("水温填补分析完成。")
    print(f"输出目录：{output_dir}")
    print(f"最终选择模型：{selected_model}")
    print(f"最终填补缺失值数量：{int(final_result['Is_Imputed'].sum())}")
    print("图件已同时保存为600 dpi PNG和可编辑PDF。")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n程序运行失败：{exc}")
        import traceback

        traceback.print_exc()
        raise