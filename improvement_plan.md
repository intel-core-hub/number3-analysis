# Numbers3プロジェクト - 改善計画と具体的なアプローチ

## 🎯 改善目標

1. **予測精度の可視化**: 機械学習モデルの精度が向上しているか確認できるようにする
2. **アプリの改善**: Streamlitアプリで簡潔な予測結果、.ipynbで詳細な分析結果を確認できるようにする
3. **ファイル構成の整理**: 役割を明確にして保守性を向上させる

---

## 📋 改善計画

### フェーズ1: 精度可視化システムの構築【最優先】

#### 1-1. 機械学習モデルの精度追跡システム
**目的**: LightGBMモデルの精度が向上しているか定量的に確認

**実装内容**:

1. **精度メトリクスの記録**

   - バックテスト時の以下の指標を記録:
     - 的中率（ストレート、ボックス）
     - Top-K精度（上位10件、20件、50件に入る確率）
     - Log-Loss（各桁の予測確率の対数損失）
     - ROI（投資収益率）
   - 日付・回号付きでCSVに保存

2. **精度推移の可視化**

   - Jupyter Notebookで以下のグラフを作成:
     - 的中率の推移（時系列グラフ）
     - Top-K精度の推移
     - ROIの累積グラフ
     - 各桁のLog-Loss推移

3. **特徴量重要度の分析**

   - LightGBMの`feature_importance`を取得
   - どの特徴量が予測に寄与しているか可視化
   - 重要度の低い特徴量を削除して精度向上を図る

**成果物**:

- `ml_performance_tracking.csv`: 精度履歴を記録するCSV
- `analysis_ml_performance.ipynb`: 精度分析用ノートブック
- `ml_feature_importance.png`: 特徴量重要度の可視化

**具体的な実装手順**:

```python

# 1. Numbers3MLBacktesterクラスを拡張

# - run()メソッドでLog-Lossを計算

# - 結果をml_performance_tracking.csvに追記

# 2. analysis_ml_performance.ipynbを作成

# - データ読み込み

# - 精度推移グラフの作成

# - 特徴量重要度の可視化

# - 各種統計分析（平均的中率、最大連続外れ回数など）

# 3. ml_backtest_runner.pyを拡張

# - 実行時に精度メトリクスを自動記録

# - --track-performance フラグで有効化
```

---

#### 1-2. ルールベースモデルとの比較分析
**目的**: 機械学習とルールベースのどちらが優れているか判定

**実装内容**:

1. **共通バックテスト環境の構築**

   - 同じ期間で両モデルをバックテスト
   - 結果を並べて比較表示

2. **比較レポートの自動生成**

    - Jupyter Notebookで以下を比較:
       - 的中率
       - ROI
       - 連続的中回数
       - 最大損失額

**成果物**:

- `model_comparison.ipynb`: モデル比較分析ノートブック
- `model_comparison_report.csv`: 比較結果サマリー

---

### フェーズ2: アプリケーションの改善

#### 2-1. Streamlitアプリの改善
**目的**: 簡潔で分かりやすい予測結果の表示

**改善内容**:

1. **予測タブの改善**

    - 予測番号に加えて以下を表示:
       - **スコア**: 総合スコア（0-100の正規化された値）
       - **各桁の確率**: 各桁が選ばれた確率（機械学習の場合）
       - **根拠**: 簡潔な1-2行の説明（例: "直近20回で頻出、合計値が平均的"）
   
2. **機械学習予測の統合**

    - 新しいタブ「AI予測（LightGBM）」を追加
    - 以下を表示:
       - 予測番号
       - 各桁の予測確率（Top3）
       - モデルの信頼度スコア
       - 過去の的中率

3. **ダッシュボード機能の追加**

    - トップページに以下のサマリーを表示:
       - 最新の予測結果（ルールベース、機械学習）
       - 過去1ヶ月の的中率
       - ROI推移グラフ（ミニグラフ）

**実装例**:
```python
# app.pyに以下を追加

# 新しいタブ: AI予測
with tab_ai_prediction:
    st.header("AI予測（LightGBM）")
    
    if st.button("AI予測実行"):
        ml_predictor = Numbers3MLPredictor(df)
        ml_predictor.train()
        prediction = ml_predictor.predict_next()
        
        # 各桁の確率を取得
        probas = {}
        for digit in ["n1", "n2", "n3"]:
            proba = ml_predictor.models[digit].predict_proba(X_test)
            probas[digit] = proba[0]
        
        st.success(f"予測番号: {prediction}")
        
        # 各桁のTop3確率を表示
        col1, col2, col3 = st.columns(3)
        for i, (digit, col) in enumerate(zip(["n1", "n2", "n3"], [col1, col2, col3])):
            col.metric(f"{['百', '十', '一'][i]}の位", prediction[i])
            top3 = np.argsort(probas[digit])[-3:][::-1]
            col.write("確率Top3:")
            for rank, idx in enumerate(top3, 1):
                col.write(f"{rank}. {idx}桁: {probas[digit][idx]:.2%}")
```

---

#### 2-2. Jupyter Notebookの詳細分析環境
**目的**: 製作者が詳細なデータ分析結果を確認できる環境を構築

**実装内容**:

1. **analysis_detailed.ipynb**: 包括的なデータ分析ノートブック

   - セクション構成:

     1. データ概要（件数、期間、欠損値）
     2. 基本統計（各桁の出現頻度、合計値の分布）
     3. 時系列分析（トレンド、季節性）
     4. パターン分析（遷移確率、ハマリ傾向）
     5. 予測モデル分析

        - ルールベースモデルの精度評価
        - 機械学習モデルの精度評価
        - 特徴量重要度

     6. バックテスト結果の詳細

        - 各モデルの的中率推移
        - ROI分析
        - 連続的中・外れの分析

2. **analysis_ml_performance.ipynb**: 機械学習特化の分析ノートブック

   - セクション構成:

     1. モデル概要と設定
     2. 学習履歴の可視化
     3. 精度メトリクスの推移
     4. 特徴量エンジニアリングの評価
     5. ハイパーパラメータの影響分析
     6. 誤分類分析（どのパターンで外れやすいか）

3. **analysis_comparison.ipynb**: モデル比較ノートブック

   - ルールベース vs 機械学習の詳細比較
   - 各モデルの得意・不得意パターンの分析
   - アンサンブル（両方を組み合わせる）の可能性検討

**成果物**:

- 上記3つのJupyter Notebookファイル
- 各種可視化グラフ（PNG保存）
- 分析結果サマリー（Markdown形式）

---

### フェーズ3: コード整理とリファクタリング

#### 3-1. ファイル分割
**目的**: `numbers3_logic.py`が肥大化しているため、役割ごとに分割

**新しいファイル構成**:
```text
project/
├── app.py                          # Streamlitアプリ
├── auto_update.py                  # 自動更新スクリプト
├── ml_backtest_runner.py           # MLバックテスト実行
├── data/
│   └── numbers3_clean.csv          # データファイル
├── results/
│   ├── ml_backtest_results.csv     # MLバックテスト結果
│   └── ml_performance_tracking.csv # ML精度履歴
├── notebooks/
│   ├── analysis_detailed.ipynb     # 詳細分析
│   ├── analysis_ml_performance.ipynb # ML精度分析
│   └── analysis_comparison.ipynb   # モデル比較
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py             # データ取得・更新
│   ├── feature_engineering.py      # 特徴量エンジニアリング
│   ├── predictors/
│   │   ├── __init__.py
│   │   ├── rule_based.py           # ルールベース予測
│   │   └── ml_predictor.py         # 機械学習予測
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── rule_backtest.py        # ルールベースバックテスト
│   │   └── ml_backtest.py          # MLバックテスト
│   └── analyzers/
│       ├── __init__.py
│       ├── interval_analyzer.py    # インターバル分析
│       ├── trend_analyzer.py       # トレンド分析
│       └── pattern_analyzer.py     # パターン分析
└── tests/
    ├── test_data_fetcher.py
    ├── test_feature_engineering.py
    ├── test_predictors.py
    └── test_backtesting.py
```

**実装手順**:

1. 現在の`numbers3_logic.py`をバックアップ
2. 各クラス・関数を適切なファイルに移動
3. インポート関係を整理
4. テストを実行して動作確認
5. `app.py`と`auto_update.py`のインポートを修正

---

### フェーズ4: 高度な機能追加

#### 4-1. アンサンブル予測
**目的**: 複数モデルを組み合わせて精度向上

**実装内容**:

- ルールベース複数モデルの投票（majority voting）
- 機械学習とルールベースの重み付き平均
- スタッキング（メタモデル）

#### 4-2. リアルタイム精度モニタリング
**目的**: 予測精度をリアルタイムで監視

**実装内容**:

- Streamlitダッシュボードに精度推移グラフを追加
- アラート機能（精度が閾値を下回ったら通知）

#### 4-3. 自動再学習
**目的**: データが更新されたら自動的にモデルを再学習

**実装内容**:

- `auto_update.py`に再学習機能を追加
- 一定期間ごとにハイパーパラメータチューニング

---

## 🛠️ 具体的な実装順序

### ステップ1: 精度追跡システムの構築（1-2日）

1. `ml_performance_tracking.csv`の設計
2. `Numbers3MLBacktester`の拡張
3. `analysis_ml_performance.ipynb`の作成
4. 既存のバックテスト結果を記録

### ステップ2: 詳細分析ノートブックの作成（1-2日）

1. `analysis_detailed.ipynb`の作成
2. データ探索と基本統計
3. 予測モデルの精度評価
4. 可視化グラフの作成

### ステップ3: Streamlitアプリの改善（1-2日）

1. 機械学習予測タブの追加
2. 予測根拠の表示機能
3. ダッシュボード機能の追加

### ステップ4: モデル比較分析（1日）

1. `analysis_comparison.ipynb`の作成
2. ルールベース vs 機械学習の比較
3. アンサンブルの可能性検討

### ステップ5: コード整理（1-2日）

1. ファイル分割計画の確定
2. リファクタリング実行
3. テスト実行と動作確認

---

## 📊 期待される成果

### 短期的な成果（1週間以内）

- 機械学習モデルの精度が可視化され、向上しているか判断できる
- Jupyter Notebookで詳細な分析結果を確認できる
- Streamlitアプリで予測根拠が分かるようになる

### 中期的な成果（2-4週間）

- コードが整理され、保守性が向上
- 複数モデルの比較が容易になる
- アンサンブル予測で精度向上

### 長期的な成果（1-3ヶ月）

- 自動再学習により常に最新のパターンを学習
- リアルタイム精度モニタリングで異常検知
- 安定した予測精度の維持

---

## ⚠️ 注意事項

1. **予測精度の限界**

   - Numbers3はランダム性が高いため、100%の的中は不可能
   - 機械学習で精度が劇的に向上するとは限らない
   - 長期的なROIが正になることを目標にする

2. **過学習のリスク**

   - 過去データに過度に適合すると、未来の予測精度が下がる
   - 定期的にバックテストで検証する

3. **データの質**

   - スクレイピングデータに欠損や誤りがないか常に確認
   - データ更新時のバリデーションを徹底

---

## 🚀 次のアクション

### 優先度：高

1. ✅ `ml_performance_tracking.csv`の設計と実装
2. ✅ `analysis_ml_performance.ipynb`の作成
3. ✅ 既存のMLバックテスト結果を詳細に分析

### 優先度：中

4. `analysis_detailed.ipynb`の作成
5. Streamlitアプリに機械学習予測タブを追加
6. モデル比較分析ノートブックの作成

### 優先度：低

7. コードのリファクタリング
8. アンサンブル予測の実装
9. 自動再学習機能の追加

---

まずは**精度追跡システムの構築**から始めることをお勧めします。これにより、機械学習モデルが実際に精度向上しているかを定量的に評価できるようになります。
