"""
src.analysis.visualizer — 可視化 (Matplotlib / Seaborn / Plotly)

責務:
    - インターバル分析 (Numbers3IntervalAnalyzer)
    - トレンド分析 (Numbers3TrendAnalyzer)
    - パターン分析 (Numbers3PatternAnalyzer)
    - 日本語フォント設定 (configure_japanese_fonts)
    - 特徴量重要度の棒グラフ
    - Plotly 累積損益チャート (Prompt C)
    - ヒットミス ヒートマップ (Prompt C)
    - 誤差分布ヒストグラム (Prompt C)
    - 曜日・パターン正確度テスト (Prompt C)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

from src.data.loader import normalize_numbers3_columns


# =====================================================================
# 日本語フォント
# =====================================================================


def configure_japanese_fonts(preferred: Optional[List[str]] = None) -> Optional[str]:
    """Matplotlib で日本語を表示できるようフォントを設定する."""
    font_path = "ipaexg.ttf"

    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
        print(f"Font loaded: {prop.get_name()}")
        return prop.get_name()
    else:
        plt.rcParams["font.family"] = "sans-serif"

    if preferred is None:
        preferred = [
            "IPAexGothic",
            "IPAexMincho",
            "Noto Sans CJK JP",
            "Yu Gothic",
            "Meiryo",
            "MS Gothic",
        ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            return name
    mpl.rcParams["axes.unicode_minus"] = False
    return None


# =====================================================================
# インターバル分析
# =====================================================================


class Numbers3IntervalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_digit_columns()
        if "回号" in self.df.columns:
            self.df["回号"] = pd.to_numeric(self.df["回号"], errors="coerce")
        self.df = self.df.sort_values("回号").copy()

    def _ensure_digit_columns(self) -> None:
        if "digit_h" in self.df.columns:
            return
        if "当選番号" not in self.df.columns:
            if "当せん番号" in self.df.columns:
                self.df["当選番号"] = self.df["当せん番号"]
            else:
                raise KeyError("当選番号が存在しないため、digit_hを作成できません。")
        num = self.df["当選番号"].astype(str).str.zfill(3)
        self.df["digit_h"] = num.str[0].astype(int)
        self.df["digit_t"] = num.str[1].astype(int)
        self.df["digit_o"] = num.str[2].astype(int)

    def _intervals(self, column: str = "digit_h") -> Dict[int, np.ndarray]:
        intervals: Dict[int, list] = {i: [] for i in range(10)}
        for num in range(10):
            appearance_indices = (
                self.df[self.df[column] == num]["回号"].dropna().values
            )
            if len(appearance_indices) > 1:
                intervals[num] = np.diff(appearance_indices)
        return intervals

    def interval_hist_fig(self, column: str = "digit_h") -> Optional[plt.Figure]:
        intervals_dict = self._intervals(column)
        all_intervals = [
            item for sublist in intervals_dict.values() for item in sublist
        ]
        if len(all_intervals) == 0:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(all_intervals, bins=50, kde=True, color="skyblue", ax=ax)
        ax.axvline(
            np.mean(all_intervals),
            color="red",
            linestyle="--",
            label=f"平均: {np.mean(all_intervals):.1f}回",
        )
        ax.set_title(f"数字出現インターバルの分布 ({column})")
        ax.set_xlabel("ハマリ回数（回）")
        ax.set_ylabel("頻度")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        return fig

    def interval_cdf_fig(self, column: str = "digit_h") -> Optional[plt.Figure]:
        intervals_dict = self._intervals(column)
        all_intervals = [
            item for sublist in intervals_dict.values() for item in sublist
        ]
        if len(all_intervals) == 0:
            return None

        sorted_intervals = np.sort(all_intervals)
        yvals = np.arange(len(sorted_intervals)) / float(
            len(sorted_intervals) - 1
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            sorted_intervals,
            yvals,
            marker=".",
            linestyle="none",
            color="navy",
        )
        ax.axhline(0.90, color="orange", linestyle="--", label="90% 累積ライン")
        ax.set_title(f"ハマリ回数の累積分布 ({column})")
        ax.set_xlabel("ハマリ回数（回）")
        ax.set_ylabel("累積確率")

        idx90 = np.where(yvals >= 0.90)[0][0]
        val90 = sorted_intervals[idx90]
        ax.axvline(val90, color="orange", linestyle="--")
        ax.annotate(
            f"90%が{val90}回以内に再出現",
            xy=(val90, 0.5),
            color="darkorange",
            fontweight="bold",
        )

        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig


# =====================================================================
# トレンド分析
# =====================================================================


class Numbers3TrendAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_sum_features()
        if "回号" in self.df.columns:
            self.df["回号"] = pd.to_numeric(self.df["回号"], errors="coerce")
        self.df = self.df.sort_values("回号").dropna(subset=["回号"]).copy()

    def _ensure_sum_features(self) -> None:
        if "digit_sum" in self.df.columns:
            return
        if "当選番号" not in self.df.columns:
            if "当せん番号" in self.df.columns:
                self.df["当選番号"] = self.df["当せん番号"]
            else:
                raise KeyError("当選番号が存在しないため、digit_sumを作成できません。")
        num = self.df["当選番号"].astype(str).str.zfill(3)
        self.df["digit_h"] = num.str[0].astype(int)
        self.df["digit_t"] = num.str[1].astype(int)
        self.df["digit_o"] = num.str[2].astype(int)
        self.df["digit_sum"] = self.df[
            ["digit_h", "digit_t", "digit_o"]
        ].sum(axis=1)

    def _theoretical_probs(self) -> np.ndarray:
        p = np.ones(10) / 10.0
        p_sum = np.convolve(p, p)
        p_sum = np.convolve(p_sum, p)
        return p_sum

    def sum_distribution_fig(self) -> plt.Figure:
        sums = self.df["digit_sum"]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(
            sums,
            bins=np.arange(29) - 0.5,
            kde=False,
            stat="density",
            color="green",
            alpha=0.6,
            label="実際の出現頻度",
            ax=ax,
        )

        theoretical_probs = self._theoretical_probs()
        x = np.arange(28)
        ax.plot(
            x,
            theoretical_probs,
            "r-o",
            linewidth=2,
            markersize=5,
            label="理論上の確率 (厳密解)",
        )

        ax.set_title("当選番号合計値の分布分析")
        ax.set_xlabel("合計値 (0〜27)")
        ax.set_ylabel("確率")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(0, 28, 2))
        return fig

    def moving_average_fig(self, window: int = 50) -> plt.Figure:
        ma = self.df["digit_sum"].rolling(window=window).mean()

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(
            self.df["回号"],
            ma,
            color="blue",
            linewidth=1,
            label=f"{window}回移動平均",
        )
        ax.axhline(
            13.5, color="red", linestyle="--", label="理論的な期待値 (13.5)"
        )

        valid_indices = ~np.isnan(ma)
        x_valid = self.df["回号"][valid_indices]
        y_valid = ma[valid_indices]

        ax.fill_between(
            x_valid, y_valid, 13.5, where=(y_valid > 13.5), color="red", alpha=0.1
        )
        ax.fill_between(
            x_valid,
            y_valid,
            13.5,
            where=(y_valid < 13.5),
            color="blue",
            alpha=0.1,
        )

        ax.set_title(f"合計値の移動平均トレンド ({window}回移動平均)")
        ax.set_xlabel("回号")
        ax.set_ylabel("合計値の平均")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        return fig


# =====================================================================
# パターン分析
# =====================================================================


class Numbers3PatternAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_pattern_features()
        if "回号" in self.df.columns:
            self.df["回号"] = pd.to_numeric(self.df["回号"], errors="coerce")
        self.df = self.df.sort_values("回号").dropna(subset=["回号"]).copy()

    def _ensure_pattern_features(self) -> None:
        if "odd_count" in self.df.columns and "big_count" in self.df.columns:
            return
        if "当選番号" not in self.df.columns:
            if "当せん番号" in self.df.columns:
                self.df["当選番号"] = self.df["当せん番号"]
            else:
                raise KeyError(
                    "当選番号が存在しないため、odd_countを作成できません。"
                )
        num = self.df["当選番号"].astype(str).str.zfill(3)
        self.df["digit_h"] = num.str[0].astype(int)
        self.df["digit_t"] = num.str[1].astype(int)
        self.df["digit_o"] = num.str[2].astype(int)
        self.df["odd_count"] = (
            (self.df[["digit_h", "digit_t", "digit_o"]] % 2 == 1).sum(axis=1)
        )
        self.df["big_count"] = (
            (self.df[["digit_h", "digit_t", "digit_o"]] >= 5).sum(axis=1)
        )

    def transition_heatmap_fig(
        self, column: str = "odd_count", title: Optional[str] = None
    ) -> plt.Figure:
        data = pd.DataFrame(
            {
                "current": self.df[column],
                "next": self.df[column].shift(-1),
            }
        ).dropna()

        cross_tab = pd.crosstab(data["current"], data["next"], normalize="index")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cross_tab, annot=True, cmap="Blues", fmt=".2%", ax=ax)

        display_title = (
            title if title else f"{column} の遷移確率（現在 -> 次回）"
        )
        ax.set_title(display_title)
        ax.set_xlabel("次回の状態")
        ax.set_ylabel("現在の状態")
        return fig


# =====================================================================
# 特徴量重要度の可視化
# =====================================================================


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "上位特徴量の重要度 (LightGBM)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    configure_japanese_fonts()

    avg_imp = (
        importance_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    avg_df = avg_imp.reset_index()
    avg_df.columns = ["feature", "importance"]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    sns.barplot(
        data=avg_df, x="importance", y="feature",
        hue="feature", dodge=False, legend=False,
        palette="viridis", ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("平均重要度 (Gain)")
    ax.set_ylabel("特徴量")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importance_per_digit(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    configure_japanese_fonts()

    digit_labels = {"n1": "百の位", "n2": "十の位", "n3": "一の位"}
    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, top_n * 0.35)))

    for idx, (digit, label) in enumerate(digit_labels.items()):
        subset = importance_df[importance_df["digit"] == digit]
        top = subset.nlargest(top_n, "importance")
        sns.barplot(
            x="importance",
            y="feature",
            data=top,
            ax=axes[idx],
            palette="coolwarm",
        )
        axes[idx].set_title(f"{label} ({digit})")
        axes[idx].set_xlabel("重要度")
        axes[idx].set_ylabel("")
        axes[idx].grid(axis="x", alpha=0.3)

    plt.suptitle("桁別 特徴量重要度", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_accuracy_trend(
    tracking_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    configure_japanese_fonts()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    if "cumulative_profit" in tracking_df.columns:
        axes[0].plot(
            tracking_df.index,
            tracking_df["cumulative_profit"],
            color="green",
            linewidth=1.2,
        )
        axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title("累積損益の推移")
        axes[0].set_ylabel("損益 (円)")
        axes[0].grid(alpha=0.3)

    if "logloss_avg" in tracking_df.columns:
        axes[1].plot(
            tracking_df.index,
            tracking_df["logloss_avg"],
            color="navy",
            linewidth=1.0,
            label="モデル Log-Loss",
        )
        random_ll = -np.log(0.1)
        axes[1].axhline(
            random_ll,
            color="red",
            linestyle="--",
            label=f"ランダム Log-Loss ({random_ll:.3f})",
        )
        axes[1].set_title("平均 Log-Loss の推移")
        axes[1].set_ylabel("Log-Loss")
        axes[1].set_xlabel("バックテスト回数")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =====================================================================
# Plotly 可視化 (Prompt C)
# =====================================================================


def plot_cumulative_profit_plotly(
    sim_df: pd.DataFrame,
    title: str = "累積損益の推移",
):
    """Plotly で累積損益のインタラクティブ折れ線チャートを作成する.

    Parameters
    ----------
    sim_df : DataFrame
        simulate_revenue() の出力 (cumulative_profit 列を含む)
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly が必要です: pip install plotly")

    fig = go.Figure()

    if "strategy" in sim_df.columns:
        for strategy_name, grp in sim_df.groupby("strategy"):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(grp))),
                    y=grp["cumulative_profit"],
                    mode="lines",
                    name=str(strategy_name),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sim_df))),
                y=sim_df["cumulative_profit"],
                mode="lines",
                name="累積損益",
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="ラウンド",
        yaxis_title="損益 (円)",
        hovermode="x unified",
    )
    return fig


def plot_hit_miss_heatmap_plotly(
    result_df: pd.DataFrame,
    title: str = "桁別ヒットミス ヒートマップ",
):
    """Plotly で桁ごとの正解/不正解をヒートマップ表示する.

    Parameters
    ----------
    result_df : DataFrame
        actual, set_pick 列を含むバックテスト結果

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly が必要です: pip install plotly")

    digits_data: List[List[int]] = []
    labels = ["百の位", "十の位", "一の位"]

    for i in range(3):
        hits = []
        for _, row in result_df.iterrows():
            actual = str(row.get("actual", "")).zfill(3)
            predicted = str(row.get("set_pick", "")).zfill(3)
            hits.append(1 if actual[i] == predicted[i] else 0)
        digits_data.append(hits)

    z = np.array(digits_data)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            y=labels,
            colorscale=[[0, "rgb(255,200,200)"], [1, "rgb(100,200,100)"]],
            showscale=True,
            colorbar=dict(title="Hit"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="ラウンド (時系列)",
        yaxis_title="桁",
    )
    return fig


def plot_error_distribution_plotly(
    result_df: pd.DataFrame,
    title: str = "予測誤差の分布",
):
    """予測と実績の差分分布を Plotly ヒストグラムで表示する.

    Parameters
    ----------
    result_df : DataFrame
        actual, set_pick 列を含むバックテスト結果

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly が必要です: pip install plotly")

    fig = make_subplots(rows=1, cols=3, subplot_titles=["百の位", "十の位", "一の位"])

    for i in range(3):
        errors = []
        for _, row in result_df.iterrows():
            actual = str(row.get("actual", "")).zfill(3)
            predicted = str(row.get("set_pick", "")).zfill(3)
            errors.append(int(actual[i]) - int(predicted[i]))

        fig.add_trace(
            go.Histogram(x=errors, name=f"桁{i+1}", nbinsx=19),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title_text=title,
        showlegend=False,
    )
    return fig


def plot_weekday_accuracy(
    result_df: pd.DataFrame,
    df_full: pd.DataFrame,
    title: str = "曜日別正確度",
):
    """曜日別の正確度を棒グラフで表示する (Prompt C).

    Parameters
    ----------
    result_df : DataFrame
        actual, set_pick, target_round 列を含む
    df_full : DataFrame
        全体データ (抽せん日列を含む)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly が必要です: pip install plotly")

    merged = result_df.copy()
    df_ref = df_full.copy()
    if "回号" in df_ref.columns:
        df_ref["回号"] = pd.to_numeric(df_ref["回号"], errors="coerce")

    if "抽せん日" in df_ref.columns:
        df_ref["dt"] = pd.to_datetime(df_ref["抽せん日"], errors="coerce")
        df_ref["weekday"] = df_ref["dt"].dt.day_name()

        round_to_weekday = dict(zip(df_ref["回号"], df_ref["weekday"]))
        if "target_round" in merged.columns:
            merged["weekday"] = merged["target_round"].map(round_to_weekday)

    if "weekday" not in merged.columns:
        # フォールバック: インデックスベースで曜日を割り当て
        merged["weekday"] = "unknown"

    merged["exact_hit"] = merged.apply(
        lambda r: 1 if str(r.get("actual", "")).zfill(3) == str(r.get("set_pick", "")).zfill(3) else 0,
        axis=1,
    )
    merged["box_hit"] = merged.apply(
        lambda r: 1
        if "".join(sorted(str(r.get("actual", "")).zfill(3)))
        == "".join(sorted(str(r.get("set_pick", "")).zfill(3)))
        else 0,
        axis=1,
    )

    weekday_stats = merged.groupby("weekday").agg(
        total=("exact_hit", "count"),
        exact_hits=("exact_hit", "sum"),
        box_hits=("box_hit", "sum"),
    ).reset_index()

    weekday_stats["exact_rate"] = weekday_stats["exact_hits"] / weekday_stats["total"] * 100
    weekday_stats["box_rate"] = weekday_stats["box_hits"] / weekday_stats["total"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weekday_stats["weekday"],
            y=weekday_stats["exact_rate"],
            name="ストレート的中率 (%)",
        )
    )
    fig.add_trace(
        go.Bar(
            x=weekday_stats["weekday"],
            y=weekday_stats["box_rate"],
            name="ボックス的中率 (%)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="曜日",
        yaxis_title="的中率 (%)",
        barmode="group",
    )
    return fig


def run_pattern_accuracy_test(
    result_df: pd.DataFrame,
) -> Dict[str, Any]:
    """パターン正確度の統計検定 (Prompt C).

    scipy binomtest を使って、各桁の正答率がランダム (10%) より
    有意に高いかを検定する。

    Returns
    -------
    Dict  各桁の正答率, p値, 判定
    """
    try:
        from scipy.stats import binomtest
    except ImportError:
        return {"error": "scipy が必要です: pip install scipy"}

    results: Dict[str, Any] = {}
    n = len(result_df)

    for i, name in enumerate(["百の位", "十の位", "一の位"]):
        hits = 0
        for _, row in result_df.iterrows():
            actual = str(row.get("actual", "")).zfill(3)
            predicted = str(row.get("set_pick", "")).zfill(3)
            if actual[i] == predicted[i]:
                hits += 1
        rate = hits / max(n, 1)
        binom_result = binomtest(hits, n, 0.1, alternative="greater")
        results[name] = {
            "hits": hits,
            "total": n,
            "rate": round(rate * 100, 2),
            "p_value": round(float(binom_result.pvalue), 4),
            "significant": binom_result.pvalue < 0.05,
        }

    return results
