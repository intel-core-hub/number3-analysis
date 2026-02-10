# Numbers3 Update Automation

ナンバーズ3の当選データを自動更新し、複数の予測モデル（ルールベース／機械学習）による次回予測を生成・可視化するプロジェクトです。

本プロジェクトでは **役割分離** を重視し、「自動更新」「通知」「結果閲覧」「分析」を明確に分けています。

---

## 🔧 システム全体の役割分離

### ① Streamlit の役割（結果を見る場所）

#### やること（Streamlit）

- 予測結果を表示
- 最新予測を強調表示
- 過去の予測履歴を一覧表示

#### やらないこと（Streamlit）

- 学習処理
- 特徴量設計
- バックテスト計算
- モデル改善

👉 **Streamlit = Viewer（閲覧専用）**

実行方法：

```bash
streamlit run streamlit_app.py
```

---

### ② Slack の役割（通知）

#### やること（Slack）

- データ更新完了通知
- 最新予測の要点のみ通知
- エラー発生時の通知

#### やらないこと（Slack）

- 詳細な分析
- 履歴管理

👉 **Slack = Push通知**

Slack通知は `auto_update.py` 内で送信されます。

---

### ③ VSCode（ローカル開発）の役割

#### やること（VSCode）

- モデル改善
- 特徴量追加
- 精度検証
- 仮説検証・試行錯誤

#### やらないこと（VSCode）

- 本番UIの表示
- 自動実行・定期処理

👉 **VSCode = 開発者の思考空間**

分析や実験は主に `notebooks/` や個別スクリプトで行います。

---

## 📁 ディレクトリ構成

```text
number3-analysis/
├ auto_update.py          # 自動更新・予測生成・Slack通知
├ numbers3_logic.py       # ロジック・モデル・ML
├ streamlit_app.py        # 結果閲覧専用UI（Streamlit）
│
├ results/                # ★ 唯一のデータ受け渡し点
│   ├ latest_prediction.csv
│   └ lightgbm_predictions.csv
│
├ notebooks/              # VSCode用・分析ノート
├ scripts/                # 補助・一時スクリプト
├ .github/workflows/      # GitHub Actions
└ README.md
```

### 設計上の重要ポイント

- **Streamlit は `results/` しか読みません**
- **`auto_update.py` は `results/` にのみ書き込みます**

👉 データフローが一方向になり、実装の見通しがよくなり、バグが大幅に減ります。

---

## ⚙ Auto update script

プロジェクトルートで以下を実行すると、データ更新・予測生成・Slack通知が行われます。

```bash
python auto_update.py
```

---

## 🤖 GitHub Actions

`.github/workflows/update_data.yml` により、以下が自動実行されます。

- 依存関係のインストール（`requirements.txt`）
- `auto_update.py` の実行
- `numbers3_clean.csv` の更新があれば自動コミット

スケジュール実行および手動実行の両方に対応しています。

---

## 📊 ML backtest（開発者向け）

機械学習モデルのバックテストはローカルで実行します。

```bash
python ml_backtest_runner.py --rounds 50 --window 300
```

定期的なハイパーパラメータ調整を行う場合（低速）：

```bash
python ml_backtest_runner.py --rounds 30 --tune-every 5 --valid-size 120
```

※ これらは **Streamlit や自動更新とは独立した開発者向け機能**です。
