# 2024年度 人工知能学会全国大会 (第38回)

## 論理制約を考慮したテーブルデータを対象とした予測モデル評価フレームワーク


## 目次

1. 背景
2. 提案手法
3. 実験
4. 結論・展望

---

## 1. 背景

- 機械学習モデルの実応用では、予測精度と信頼性の両立が重要
- 予測モデルに背景知識を制約として組み込むことで信頼性を確保

### 課題

- 論理制約付き予測モデルの評価が難しい
- 手動で設定する論理制約の恣意性

---

## 2. 提案手法

### フレームワーク概要

1. 論理制約の自動抽出
2. 制約論理式の構築
3. 予測モデルの学習
4. 予測モデルの評価

### 論理制約の自動抽出

- テーブルデータを離散化し、RuleFitによりルールを抽出

### 制約論理式の構築

- 抽出されたルールを基に制約論理式を生成

### 予測モデルの学習

- 制約条件を含む予測モデルを学習

### 予測モデルの評価

- 予測精度と論理制約の充足率を評価

---

## 3. 実験

### 実験設定

- データセット: Pima Indian Diabetes
- タスク: 二値分類
- 評価指標: ROC-AUC、論理制約の充足率

### 実験内容

1. 予測モデル間の比較
2. 論理制約採用の閾値
3. 自動抽出手法の比較 (RuleFit vs Association Rule)

### 結果

- r-SVM-pとLogReg-pは制約無しモデルに近い予測精度を示し、充足率で優位性を示した
- RuleFitはAssociation Ruleと比較して高い充足率を示した

---

## 4. 結論・展望

### まとめ

- 提案フレームワークにより、多種の予測モデルに対して論理制約の評価を行った
- r-SVM-pとLogReg-pは予測性能と充足率の両立が可能であることが示された
- RuleFitはAssociation Ruleと比較して高い予測性能と制約充足率を両立できる

### 今後の課題

- 実際の問題への適用時の論理制約の妥当性の確認
- マルチタスクやマルチラベルなど、適用可能なデータ形式の拡張

---

## 参考文献

- Friedman, J. H. and Popescu, B. E. (2008). Predictive learning via rule ensembles.
- Giannini, F., Diligenti, M., Gori, M., and Maggini, M. (2017). Learning Lukasiewicz Logic Fragments by Quadratic Programming.
- Goyal, K., Dumancic, S., and Blockeel, H. (2022). Sade: Learning models that provably satisfy domain constraints.
- Kautz, H. (2022). The third AI summer: AaaIrobert S. Engelmore Memorial Lecture.
- Wang, Z., Vijayakumar, S., Lu, K., Ganesh, V., Jha, S., and Fredrikson, M. (2023). Grounding Neural Inference with Satisfiability Modulo Theories.
- Yang, Z., Lee, J., and Park, C. (2022). Injecting logical constraints into neural networks via straight-through estimators.
- Roychowdhury, S., Diligenti, M., and Gori, M. (2021). Regularizing deep networks with prior knowledge: A constraint-based approach.
