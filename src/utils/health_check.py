"""
src.utils.health_check — システムヘルスチェック

MLOps Phase 25: システム監視と稼働状態診断

機能:
    - システム稼働時間のトラッキング
    - ディスク容量の監視
    - データベース整合性の確認
    - 直近のスクレイピング成功率
    - モデル更新頻度とドリフトスコアの推移
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import DB_PATH, RESULTS_DIR
from src.database import Numbers3Database
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """ヘルスチェック結果"""
    timestamp: str
    overall_status: str  # "healthy", "warning", "critical"
    uptime_days: int
    disk_usage_percent: float
    database_records: int
    database_last_update: str | None
    scraping_success_rate: float  # Last 30 days
    model_champion_version: str | None
    model_champion_roi: float
    model_last_update: str | None
    drift_score_last: float
    drift_score_avg_30d: float
    issues: list[str]


class HealthCheck:
    """システムヘルスチェッカー"""

    def __init__(
        self,
        db_path: Path = Path(DB_PATH),
        results_dir: Path = Path(RESULTS_DIR),
        registry: ModelRegistry | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.results_dir = Path(results_dir)
        self.registry = registry or ModelRegistry()
        self.health_log_path = self.results_dir / "health_check.json"

    def check_health(self) -> HealthStatus:
        """
        システム全体のヘルスチェックを実行

        Returns:
            HealthStatus
        """
        issues = []
        timestamp = datetime.now().isoformat()

        # 1. システム稼働時間
        uptime_days = self._get_uptime_days()
        if uptime_days > 90:
            issues.append("System running for over 90 days - consider restarting")

        # 2. ディスク容量
        disk_usage = self._get_disk_usage()
        if disk_usage > 90.0:
            issues.append(f"High disk usage: {disk_usage:.1f}%")
        elif disk_usage > 80.0:
            issues.append(f"Warning: disk usage at {disk_usage:.1f}%")

        # 3. データベース状態
        db_records, db_last_update = self._check_database()
        if db_records == 0:
            issues.append("Database is empty")
        elif db_last_update:
            last_update_dt = datetime.fromisoformat(db_last_update)
            days_since_update = (datetime.now() - last_update_dt).days
            if days_since_update > 7:
                issues.append(f"Database not updated for {days_since_update} days")

        # 4. スクレイピング成功率
        scraping_success_rate = self._check_scraping_success_rate()
        if scraping_success_rate < 0.8:
            issues.append(f"Low scraping success rate: {scraping_success_rate:.1%}")

        # 5. モデル状態
        champion = self.registry.get_champion()
        model_version = champion.version if champion else None
        model_roi = champion.roi if champion else 0.0
        model_last_update = champion.created_at if champion else None

        if not champion:
            issues.append("No champion model registered")
        elif model_last_update:
            last_update_dt = datetime.fromisoformat(model_last_update)
            days_since_model_update = (datetime.now() - last_update_dt).days
            if days_since_model_update > 30:
                issues.append(f"Champion model not updated for {days_since_model_update} days")

        # 6. ドリフトスコア
        drift_score_last, drift_score_avg = self._check_drift_scores()
        if drift_score_last > 0.7:
            issues.append(f"High recent drift score: {drift_score_last:.3f}")

        # 総合判定
        overall_status = self._determine_overall_status(issues, disk_usage, scraping_success_rate)

        health_status = HealthStatus(
            timestamp=timestamp,
            overall_status=overall_status,
            uptime_days=uptime_days,
            disk_usage_percent=disk_usage,
            database_records=db_records,
            database_last_update=db_last_update,
            scraping_success_rate=scraping_success_rate,
            model_champion_version=model_version,
            model_champion_roi=model_roi,
            model_last_update=model_last_update,
            drift_score_last=drift_score_last,
            drift_score_avg_30d=drift_score_avg,
            issues=issues,
        )

        # ログに保存
        self._save_health_log(health_status)

        return health_status

    def _get_uptime_days(self) -> int:
        """システム稼働時間を推定（ログファイルの最古日時から計算）"""
        try:
            log_dir = Path("logs")
            if not log_dir.exists():
                return 0

            log_files = list(log_dir.glob("*.log"))
            if not log_files:
                return 0

            oldest_file = min(log_files, key=lambda p: p.stat().st_ctime)
            oldest_time = datetime.fromtimestamp(oldest_file.stat().st_ctime)
            uptime = (datetime.now() - oldest_time).days
            return uptime
        except Exception as e:
            logger.warning("Failed to calculate uptime: %s", e)
            return 0

    def _get_disk_usage(self) -> float:
        """ディスク使用率を取得（プロジェクトディレクトリ）"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(Path.cwd())
            usage_percent = (used / total) * 100
            return usage_percent
        except Exception as e:
            logger.warning("Failed to get disk usage: %s", e)
            return 0.0

    def _check_database(self) -> tuple[int, str | None]:
        """データベースの状態をチェック"""
        try:
            if not self.db_path.exists():
                return 0, None

            db = Numbers3Database(str(self.db_path))
            df = db.load_draws()
            if df.empty:
                return 0, None

            record_count = len(df)

            # 最終更新日を取得
            if "抽せん日" in df.columns:
                last_date = pd.to_datetime(df["抽せん日"], errors="coerce").max()
                if pd.notna(last_date):
                    return record_count, last_date.isoformat()

            return record_count, None

        except Exception as e:
            logger.warning("Failed to check database: %s", e)
            return 0, None

    def _check_scraping_success_rate(self) -> float:
        """直近30日のスクレイピング成功率を推定"""
        try:
            # decision_historyから推定（予測が記録されていれば成功とみなす）
            decision_file = self.results_dir / "phase21_decision_history.csv"
            if not decision_file.exists():
                return 1.0  # データなし→成功扱い

            df = pd.read_csv(decision_file, encoding="utf-8-sig")
            if df.empty:
                return 1.0

            # 直近30日のレコードを抽出
            if "timestamp" not in df.columns:
                return 1.0

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            cutoff = datetime.now() - timedelta(days=30)
            recent = df[df["timestamp"] >= cutoff]

            if len(recent) == 0:
                return 1.0

            # 成功率を計算（予測があれば成功）
            success_count = len(recent[recent["candidates"].notna()])
            success_rate = success_count / len(recent)
            return success_rate

        except Exception as e:
            logger.warning("Failed to check scraping success rate: %s", e)
            return 1.0

    def _check_drift_scores(self) -> tuple[float, float]:
        """ドリフトスコアの推移を確認"""
        try:
            decision_file = self.results_dir / "phase21_decision_history.csv"
            if not decision_file.exists():
                return 0.0, 0.0

            df = pd.read_csv(decision_file, encoding="utf-8-sig")
            if df.empty or "drift_score" not in df.columns:
                return 0.0, 0.0

            # 最新のスコア
            drift_last = df["drift_score"].iloc[-1] if len(df) > 0 else 0.0

            # 直近30日の平均
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                cutoff = datetime.now() - timedelta(days=30)
                recent = df[df["timestamp"] >= cutoff]
                drift_avg = recent["drift_score"].mean() if len(recent) > 0 else 0.0
            else:
                drift_avg = df["drift_score"].tail(30).mean() if len(df) >= 30 else drift_last

            return float(drift_last), float(drift_avg)

        except Exception as e:
            logger.warning("Failed to check drift scores: %s", e)
            return 0.0, 0.0

    def _determine_overall_status(
        self,
        issues: list[str],
        disk_usage: float,
        scraping_success_rate: float,
    ) -> str:
        """総合的なヘルスステータスを判定"""
        critical_keywords = ["empty", "not updated", "High disk"]
        has_critical = any(
            any(keyword in issue for keyword in critical_keywords)
            for issue in issues
        )

        if has_critical or disk_usage > 95 or scraping_success_rate < 0.5:
            return "critical"
        elif len(issues) > 0 or disk_usage > 80:
            return "warning"
        else:
            return "healthy"

    def _save_health_log(self, status: HealthStatus) -> None:
        """ヘルスチェック結果をログに保存"""
        try:
            # 既存ログを読み込み
            if self.health_log_path.exists():
                with open(self.health_log_path, encoding="utf-8") as f:
                    log_data = json.load(f)
            else:
                log_data = {"history": []}

            # 新しい結果を追加
            log_data["history"].append(asdict(status))

            # 最新100件のみ保持
            log_data["history"] = log_data["history"][-100:]

            # 保存
            with open(self.health_log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            logger.info("Health check saved: %s", status.overall_status)

        except Exception as e:
            logger.error("Failed to save health log: %s", e)


def run_health_check() -> HealthStatus:
    """
    ヘルスチェックを実行して結果を返す

    Returns:
        HealthStatus
    """
    checker = HealthCheck()
    return checker.check_health()
