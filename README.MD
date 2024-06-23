# 論理制約を考慮したテーブルデータを対象とした予測モデル評価フレームワーク

旧リポジトリ URL 
https://github.com/k-onoue/lukasiewicz_1

## ディレクトリ構成

---

- LUKASIEWICZ_2
  - data ... データセット
  - experiment_manager ... 実験の実行ファイル
  - experiment_result ... 実験結果の格納
  - materials ... 論文，発表スライド等
  - notebooks ... 実装の際の簡易テスト用 jupyternotebook 
  - src ... ソースコード
    - association_rule.py ... Association Rule Mining
    - evaluation.py ... 予測結果の評価
    - misc.py ... 各種ユーティリティ
    - objective_function_single_task.py ... 双対形式用の目的関数
    - operators.py ... 論理演算子の変換
    - predicate_single_task.py ... 双対形式用の述語
    - preprocess_fol.py ... 制約論理式の処理
    - rulefit.py ... RuleFit
    - setup_problem_dual_single_task.py ... 双対形式用のメインクラス
    - setup_problem_primal.py ... 主形式用のメインクラス
  - .gitignore
  - instruction.ipynb ... プログラムの実行手順
  - README.md
  - requirements.txt

## 研究発表内容

---

[materials](https://github.com/k-onoue/lukasiewicz_2/blob/main/materials/) ディレクトリ内を参照

## プログラムの実行手順

---

[instruction.ipynb](https://github.com/k-onoue/lukasiewicz_2/blob/main/instruction.ipynb) を参照

