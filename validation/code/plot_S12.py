#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新版绘图脚本（适配新表头、4 个指标、生成单张与合并图）
说明（按用户要求修改）：
1. 适配新 CSV 表头（重复的 JDAY 列）。会自动识别 T / TN / NH4 / DO 的观测列和 W2（模拟）列。
2. 数据为 2025 年，包含水温(T)、总氮(TN)、氨氮(NH4)、溶解氧(DO)；为每个指标分别绘制：
   - 单张图：上为时间序列（观测 + 模拟 + 模拟误差条），下为散点图（Obs vs Sim，带残差密度 inset）。
   - 合并图：一张 2x4 的大图，第一行为 4 个时间序列，第二行为 4 个散点图。
3. 去掉了之前的 hexbin/colorbar 设定（不再绘制 hexbin）。
4. 输出：每个指标的单独 PNG/PDF，以及合并图 PNG/PDF；并生成统一误差报告 txt。
依赖：numpy, pandas, matplotlib, sklearn
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 新增代码：确保PDF字体在Ai中可编辑 ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# ------------------ 全局字体设置（Times New Roman，粗体） ------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})
# ------------------------------------------------------------------------

# ------------------ 配置（请根据实际修改路径） ------------------
INPUT_FILE_PATH = str(Path(__file__).resolve().parents[1]/"data/W2_2025_observed_simulated.csv")
OUTPUT_DIR = str(Path(__file__).resolve().parents[2]/"generated/validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 标色（观测/模拟）
COLORS = {
    'obs': [200/255, 62/255, 75/255],    # 观测色（深红）
    'sim': [88/255, 148/255, 200/255]    # 模拟色（蓝）
}

# 底部散点颜色：用户要求 RGB(255,211,212)
BOTTOM_SCATTER_COLOR = (255/255, 211/255, 212/255)  # normalized 0-1

BOTTOM_SCATTER_STYLE = {
    's': 30,                     # 点大小（略大）
    'alpha': 0.85,               # 透明度
    'facecolors': BOTTOM_SCATTER_COLOR,
    'edgecolors': 'black',       # 黑色细边
    'linewidths': 0.35,          # 边框线宽
    'zorder': 3
}
# ------------------------------------------------------------------------

VARIABLES = [
    ('T', 'Temperature (°C)'),
    ('TN', 'Total N'),
    ('NH4', 'NH4-N'),
    ('DO', 'Dissolved O2 (mg/L)')
]

def _bolden_tick_labels(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


def compute_pbias(obs, sim):
    """
    计算PBIAS（Percent Bias）。
    公式：PBIAS = 100 * sum(sim - obs) / sum(obs)
    返回百分比数值（例如：-5.23 表示 -5.23%）
    如果 obs 的总和为 0 或接近 0，则返回 np.nan
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if obs.size == 0:
        return np.nan
    obs_sum = np.nansum(obs)
    if np.isclose(obs_sum, 0.0):
        return np.nan
    return 100.0 * (np.nansum(sim - obs)) / obs_sum


def calculate_error_metrics(obs, sim):
    """计算误差指标（忽略 NaN），并返回包含 PBIAS 的字典"""
    mask = ~(pd.isna(obs) | pd.isna(sim))
    obs_c = obs[mask].astype(float)
    sim_c = sim[mask].astype(float)
    n = len(obs_c)
    if n == 0:
        return {'N': 0, 'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan, 'Bias': np.nan, 'MAPE': np.nan, 'PBIAS': np.nan}
    mse = mean_squared_error(obs_c, sim_c)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(obs_c, sim_c)
    try:
        r2 = r2_score(obs_c, sim_c)
    except Exception:
        r2 = np.nan
    bias = np.mean(sim_c - obs_c)
    nonzero_mask = obs_c != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((obs_c[nonzero_mask] - sim_c[nonzero_mask]) / obs_c[nonzero_mask])) * 100
    else:
        mape = np.nan
    pbias = compute_pbias(obs_c, sim_c)
    return {'N': n, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'Bias': bias, 'MAPE': mape, 'PBIAS': pbias}


def print_metrics_table(metrics_dict, out_path=None):
    """打印并保存误差报告（metrics_dict: varname -> metrics）"""
    txt_lines = []
    header = "=" * 60
    txt_lines.append(header)
    txt_lines.append("               南山 2025 各变量 模型误差分析报告")
    txt_lines.append(header)
    txt_lines.append("")

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for var, met in metrics_dict.items():
        txt_lines.append(f"{var} 指标性能:")
        for k, v in met.items():
            # 对 PBIAS 单独添加 % 符号显示
            if isinstance(v, float):
                if k == 'PBIAS':
                    if np.isfinite(v):
                        txt_lines.append(f"  {k:>6}: {v:>9.4f}%")
                    else:
                        txt_lines.append(f"  {k:>6}: {'N/A':>10}")
                else:
                    txt_lines.append(f"  {k:>6}: {v:>10.4f}")
            else:
                txt_lines.append(f"  {k:>6}: {v}")
        txt_lines.append("")

    txt_lines.append(f"报告生成时间: {ts}")
    txt = "\n".join(txt_lines)
    print(txt)
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(txt)
        print(f"已将误差报告保存至: {out_path}")


def _smooth_array(y, window=5):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    y_padded = np.pad(y, (window//2, window-1-window//2), mode='edge')
    y_smooth = np.convolve(y_padded, kernel, mode='valid')
    return y_smooth


def find_columns_by_pattern(df_columns, target):
    """
    在 columns 列表中查找包含 target 的列名（不区分大小写）。
    可能返回多个，函数会返回第一个匹配项（调用方可做更多判断）。
    """
    target = target.lower()
    for col in df_columns:
        if target in col.lower():
            return col
    return None


def parse_input_dataframe(df):
    """
    针对表头类似：
    JDAY, T_Obs, JDAY, T_W2, JDAY, TN_Obs, JDAY, TN_W2, JDAY, NH4_Obs, JDAY, NH4_W2, JDAY, DO_Obs, JDAY, DO_W2
    自动识别每个变量对应的 JDAY_obs/JDAY_sim/T_obs/T_sim 列名并返回字典。
    返回结构：
    {
      'T': {'jday_obs': colname, 'obs': colname, 'jday_sim': colname, 'sim': colname},
      ...
    }
    如果找不到某些列，使用 None 填充。
    """
    cols = list(df.columns)
    mapping = {}
    # 变量名小写匹配关键字
    for var, _ in VARIABLES:
        obs_key = f"{var}_obs".lower()
        sim_key = f"{var}_w2".lower()
        # 先直接找精确或包含名称的列
        obs_col = find_columns_by_pattern(cols, obs_key)
        sim_col = find_columns_by_pattern(cols, sim_key)
        # 找对应的 JDAY 列：尽量找位于 obs_col 左侧最近的包含 'jday' 的列；否则任意一个包含 jday 的列
        def find_jday_near(colname):
            if colname is None:
                return find_columns_by_pattern(cols, 'jday')
            try:
                idx = cols.index(colname)
            except ValueError:
                return find_columns_by_pattern(cols, 'jday')
            # search left up to 5 positions
            for i in range(idx-1, max(-1, idx-6), -1):
                if 'jday' in cols[i].lower():
                    return cols[i]
            # fallback: any jday
            return find_columns_by_pattern(cols, 'jday')
        jday_obs_col = find_jday_near(obs_col)
        jday_sim_col = find_jday_near(sim_col)
        mapping[var] = {
            'jday_obs': jday_obs_col,
            'obs': obs_col,
            'jday_sim': jday_sim_col,
            'sim': sim_col
        }
    return mapping


def safe_series(df, col):
    """返回列的 numeric Series，若 col 为 None 或不存在返回全为 NaN 的 Series（长度与 df 相同）"""
    if col is None:
        return pd.Series([np.nan] * len(df))
    if col not in df.columns:
        return pd.Series([np.nan] * len(df))
    return pd.to_numeric(df[col], errors='coerce')


def plot_single_variable(df, varname, varlabel, cols_map, save_base):
    """
    为单个变量生成一张图（上：时间序列，下：散点图），并保存 PNG/PDF。
    返回计算的误差指标（dict）。
    """
    jday_obs = safe_series(df, cols_map.get('jday_obs'))
    obs = safe_series(df, cols_map.get('obs'))
    jday_sim = safe_series(df, cols_map.get('jday_sim'))
    sim = safe_series(df, cols_map.get('sim'))

    # 计算 metrics（基于 obs 和 sim）
    metrics = calculate_error_metrics(obs, sim)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={'height_ratios': [2, 2]})
    # 时间序列
    ax = axes[0]
    ax.plot(jday_obs, obs, marker='o', markersize=4.0, linewidth=1.1, label='Observed', color=COLORS['obs'])
    mask_ts = ~(pd.isna(obs) | pd.isna(sim) | pd.isna(jday_sim))
    if mask_ts.any():
        x_sim = jday_sim[mask_ts]
        y_sim = sim[mask_ts]
        yerr = np.abs(y_sim - obs[mask_ts])
        ax.errorbar(x_sim, y_sim, yerr=yerr, fmt='s', markersize=3.6, linewidth=0.9,
                    ecolor=[0.6, 0.6, 0.6], elinewidth=0.6, alpha=0.95, label='Simulated (with error)',
                    color=COLORS['sim'], capsize=2)
        ax.plot(jday_sim, sim, marker='s', markersize=3.6, linewidth=1.0, label='_nolegend_', color=COLORS['sim'])
    else:
        ax.plot(jday_sim, sim, marker='s', markersize=3.6, linewidth=1.0, label='Simulated', color=COLORS['sim'])

    ax.set_title(f'{varlabel} - Time series (2025)', fontweight='bold')
    ax.set_xlabel('Julian Day', fontweight='bold'); ax.set_ylabel(varlabel, fontweight='bold')
    ax.legend(frameon=False, prop={'weight': 'bold'})
    _bolden_tick_labels(ax)

    # 散点图
    ax = axes[1]
    mask_sc = ~(pd.isna(obs) | pd.isna(sim))
    obs_sc = obs[mask_sc].astype(float)
    sim_sc = sim[mask_sc].astype(float)
    if len(obs_sc) > 0:
        ax.scatter(obs_sc, sim_sc,
                   s=BOTTOM_SCATTER_STYLE['s'],
                   alpha=BOTTOM_SCATTER_STYLE['alpha'],
                   facecolors=BOTTOM_SCATTER_STYLE['facecolors'],
                   edgecolors=BOTTOM_SCATTER_STYLE['edgecolors'],
                   linewidths=BOTTOM_SCATTER_STYLE['linewidths'],
                   zorder=BOTTOM_SCATTER_STYLE['zorder'])
        mn, mx = min(obs_sc.min(), sim_sc.min()), max(obs_sc.max(), sim_sc.max())
        margin = 0.03 * (mx - mn) if (mx - mn) > 0 else 0.1
        ax.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
                '--', color=[0.5, 0.5, 0.5], linewidth=1)
        ax.set_xlim(mn - margin, mx + margin); ax.set_ylim(mn - margin, mx + margin)

        # metrics 显示（包含 PBIAS）
        pbias_val = metrics.get('PBIAS', np.nan)
        pbias_str = f"{pbias_val:.2f}%" if np.isfinite(pbias_val) else "N/A"
        info = f"N={metrics['N']}\nRMSE={metrics['RMSE']:.3f}\nMAE={metrics['MAE']:.3f}\nR²={metrics['R²']:.3f}\nPBIAS={pbias_str}"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        # resid inset
        resid = (sim_sc - obs_sc)
        inset_ax = ax.inset_axes([0.62, 0.10, 0.35, 0.35])
        if len(resid) >= 3:
            counts, bins = np.histogram(resid, bins=30, density=True)
            bins_center = (bins[:-1] + bins[1:]) / 2.0
            smooth_counts = _smooth_array(counts, window=3)
            inset_ax.fill_between(bins_center, 0, smooth_counts, color=[0.85, 0.85, 0.95], alpha=0.95)
            inset_ax.plot(bins_center, smooth_counts, color=[0.15, 0.15, 0.45], linewidth=1.0)
            mu = np.mean(resid)
            sigma = np.std(resid)
            inset_ax.axvline(mu, color='k', linestyle='--', linewidth=0.8)
            inset_ax.axvline(mu - sigma, color='gray', linestyle=':', linewidth=0.7)
            inset_ax.axvline(mu + sigma, color='gray', linestyle=':', linewidth=0.7)
            inset_ax.text(0.98, 0.95, f"μ={mu:.2f}\nσ={sigma:.2f}", transform=inset_ax.transAxes,
                          fontsize=7, fontweight='bold', va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        else:
            inset_ax.hist(resid, bins=10, density=True, color=[0.85, 0.85, 0.95], edgecolor='k')
        inset_ax.set_title('Resid. density', fontsize=8, fontweight='bold')
        inset_ax.tick_params(axis='both', which='major', labelsize=7)
        _bolden_tick_labels(inset_ax)
    else:
        ax.text(0.5, 0.5, 'No paired Obs/Sim data', ha='center', va='center', fontweight='bold')
    ax.set_title(f'{varlabel} - Obs vs Sim', fontweight='bold')
    ax.set_xlabel('Observed', fontweight='bold'); ax.set_ylabel('Simulated', fontweight='bold')
    _bolden_tick_labels(ax)

    plt.tight_layout()
    png_path = save_base + '.png'
    pdf_path = save_base + '.pdf'
    fig.savefig(png_path, dpi=450, bbox_inches='tight')
    # 保存高分辨率PDF
    fig.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return metrics, png_path, pdf_path


def plot_combined(df, mappings, save_base):
    """
    生成合并图：2 行 x 4 列
    第一行：4 个时间序列（T, TN, NH4, DO）
    第二行：4 个散点图（Obs vs Sim）
    """
    fig, axes = plt.subplots(2, len(VARIABLES), figsize=(16, 8), sharex=False)
    metrics_all = {}
    for col_idx, (var, varlabel) in enumerate(VARIABLES):
        cols_map = mappings[var]
        jday_obs = safe_series(df, cols_map.get('jday_obs'))
        obs = safe_series(df, cols_map.get('obs'))
        jday_sim = safe_series(df, cols_map.get('jday_sim'))
        sim = safe_series(df, cols_map.get('sim'))

        # metrics
        metrics = calculate_error_metrics(obs, sim)
        metrics_all[var] = metrics

        # time series (top row)
        ax_ts = axes[0, col_idx]
        ax_ts.plot(jday_obs, obs, marker='o', markersize=3.5, linewidth=1.0, label='Observed', color=COLORS['obs'])
        mask_ts = ~(pd.isna(obs) | pd.isna(sim) | pd.isna(jday_sim))
        if mask_ts.any():
            x_sim = jday_sim[mask_ts]
            y_sim = sim[mask_ts]
            yerr = np.abs(y_sim - obs[mask_ts])
            # 注意：这里我们给 errorbar 设置 label，它将代表模拟数据
            ax_ts.errorbar(x_sim, y_sim, yerr=yerr, fmt='s', markersize=3.0, linewidth=0.9,
                           ecolor=[0.6, 0.6, 0.6], elinewidth=0.6, alpha=0.9, label='Simulated',
                           # <-- 标签可以简化为 'Simulated'
                           color=COLORS['sim'], capsize=2)
            # 这条线用于连接模拟点，不需要在图例中显示
            ax_ts.plot(jday_sim, sim, marker='s', markersize=3.0, linewidth=0.9, label='_nolegend_',
                       color=COLORS['sim'])
        else:
            ax_ts.plot(jday_sim, sim, marker='s', markersize=3.0, linewidth=1.0, label='Simulated', color=COLORS['sim'])

        #
        # --- 添加的代码在这里 ---
        # 调用 legend() 函数来显示图例
        # 使用 fontsize='small' 避免图例过大，影响视图
        ax_ts.legend(frameon=False, prop={'weight': 'bold'}, fontsize='small')
        # ----------------------
        #

        ax_ts.set_title(f'{varlabel} - Time series', fontweight='bold')
        ax_ts.set_xlabel('Julian Day', fontweight='bold');
        ax_ts.set_ylabel(varlabel, fontweight='bold')
        _bolden_tick_labels(ax_ts)

        # scatter (bottom row)
        ax_sc = axes[1, col_idx]
        mask_sc = ~(pd.isna(obs) | pd.isna(sim))
        obs_sc = obs[mask_sc].astype(float)
        sim_sc = sim[mask_sc].astype(float)
        if len(obs_sc) > 0:
            ax_sc.scatter(obs_sc, sim_sc,
                          s=BOTTOM_SCATTER_STYLE['s'],
                          alpha=BOTTOM_SCATTER_STYLE['alpha'],
                          facecolors=BOTTOM_SCATTER_STYLE['facecolors'],
                          edgecolors=BOTTOM_SCATTER_STYLE['edgecolors'],
                          linewidths=BOTTOM_SCATTER_STYLE['linewidths'],
                          zorder=BOTTOM_SCATTER_STYLE['zorder'])
            mn, mx = min(obs_sc.min(), sim_sc.min()), max(obs_sc.max(), sim_sc.max())
            margin = 0.03 * (mx - mn) if (mx - mn) > 0 else 0.1
            ax_sc.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
                       '--', color=[0.5, 0.5, 0.5], linewidth=1)
            ax_sc.set_xlim(mn - margin, mx + margin);
            ax_sc.set_ylim(mn - margin, mx + margin)
            pbias_val = metrics.get('PBIAS', np.nan)
            pbias_str = f"{pbias_val:.2f}%" if np.isfinite(pbias_val) else "N/A"
            info = f"N={metrics['N']}\nRMSE={metrics['RMSE']:.3f}\nMAE={metrics['MAE']:.3f}\nR²={metrics['R²']:.3f}\nPBIAS={pbias_str}"
            ax_sc.text(0.02, 0.98, info, transform=ax_sc.transAxes, fontsize=8, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            # inset
            resid = (sim_sc - obs_sc)
            inset_ax = ax_sc.inset_axes([0.62, 0.06, 0.35, 0.35])
            if len(resid) >= 3:
                counts, bins = np.histogram(resid, bins=30, density=True)
                bins_center = (bins[:-1] + bins[1:]) / 2.0
                smooth_counts = _smooth_array(counts, window=3)
                inset_ax.fill_between(bins_center, 0, smooth_counts, color=[0.85, 0.85, 0.95], alpha=0.95)
                inset_ax.plot(bins_center, smooth_counts, color=[0.15, 0.15, 0.45], linewidth=1.0)
                mu = np.mean(resid);
                sigma = np.std(resid)
                inset_ax.axvline(mu, color='k', linestyle='--', linewidth=0.8)
                inset_ax.axvline(mu - sigma, color='gray', linestyle=':', linewidth=0.7)
                inset_ax.axvline(mu + sigma, color='gray', linestyle=':', linewidth=0.7)
                inset_ax.text(0.98, 0.95, f"μ={mu:.2f}\nσ={sigma:.2f}", transform=inset_ax.transAxes,
                              fontsize=7, fontweight='bold', va='top', ha='right',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            else:
                inset_ax.hist(resid, bins=10, density=True, color=[0.85, 0.85, 0.95], edgecolor='k')
            inset_ax.set_title('Resid. density', fontsize=7, fontweight='bold')
            inset_ax.tick_params(axis='both', which='major', labelsize=7)
            _bolden_tick_labels(inset_ax)
        else:
            ax_sc.text(0.5, 0.5, 'No paired Obs/Sim data', ha='center', va='center', fontweight='bold')
        ax_sc.set_title(f'{varlabel} - Obs vs Sim', fontweight='bold')
        ax_sc.set_xlabel('Observed', fontweight='bold');
        ax_sc.set_ylabel('Simulated', fontweight='bold')
        _bolden_tick_labels(ax_sc)

    # 调整整体布局，防止标题和标签重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # 稍微调整布局为标题留出空间

    png_path = save_base + '.png'
    pdf_path = save_base + '.pdf'
    fig.savefig(png_path, dpi=450, bbox_inches='tight')
    # 保存高分辨率PDF
    fig.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return metrics_all, png_path, pdf_path


def main():
    if not os.path.exists(INPUT_FILE_PATH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE_PATH}")
    print(f"使用输入文件: {INPUT_FILE_PATH}")
    df = pd.read_csv(INPUT_FILE_PATH)

    # 识别列映射
    mappings = parse_input_dataframe(df)
    # debug 打印识别结果
    print("识别到的列映射（var: mapping）:")
    for var, m in mappings.items():
        print(f"{var}: {m}")

    # 为每个变量绘制单独图，并收集 metrics 与输出路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_metrics = {}
    saved_files = []
    for var, varlabel in VARIABLES:
        cols_map = mappings[var]
        base = os.path.join(OUTPUT_DIR, f"{var}_2025_{timestamp}")
        metrics, png_path, pdf_path = plot_single_variable(df, var, varlabel, cols_map, base)
        all_metrics[var] = metrics
        saved_files.append((png_path, pdf_path))
        print(f"{var} 图已保存: {png_path} , {pdf_path}")

    # 合并图（2x4）
    combined_base = os.path.join(OUTPUT_DIR, f"combined_2x4_2025_{timestamp}")
    metrics_all, combined_png, combined_pdf = plot_combined(df, mappings, combined_base)
    print(f"合并图已保存: {combined_png} , {combined_pdf}")

    # 保存误差报告
    report_path = os.path.join(OUTPUT_DIR, f"误差报告_2025_{timestamp}.txt")
    print_metrics_table(all_metrics, out_path=report_path)

    print("所有图与报告已生成。输出文件：")
    for p, q in saved_files:
        print(f" - {p}")
    print(f" - {combined_png}")
    print(f" - {combined_pdf}")
    print(f" - {report_path}")


if __name__ == "__main__":
    main()
