# 論理制約付き予測モデル for テーブルデータ

本プログラムは，表形式データから抽出した論理制約を組み込んだ予測モデルの構築から評価までを行うためのフレームワークです．CVXPY（凸最適化ソルバー）を基盤としています．

__リンク__

- [CVXPY GitHub リポジトリ](https://github.com/cvxpy/cvxpy)
- [RuleFit Github Repository](https://github.com/christophM/rulefit)
- [Learning Lukasiewicz Logic Fragments by Quadratic Programming [Giannini 2017]](http://ecmlpkdd2017.ijs.si/papers/paperID223.pdf)

__目次__
- [論理制約付き予測モデル for テーブルデータ](#論理制約付き予測モデル-for-テーブルデータ)
  - [1. インストール](#1-インストール)
  - [2. 基本的な使用方法](#2-基本的な使用方法)
    - [データの読み込み](#データの読み込み)
    - [論理制約の抽出](#論理制約の抽出)
    - [入力情報を辞書として設定](#入力情報を辞書として設定)
    - [予測モデルの訓練](#予測モデルの訓練)
    - [予測](#予測)
    - [予測結果の評価](#予測結果の評価)
  - [3. ディレクトリ構成](#3-ディレクトリ構成)
  - [4. 質問](#4-質問)
  - [5. プラットフォーム](#5-プラットフォーム)
  - [6. 引用](#6-引用)


## 1. インストール

リポジトリのクローンと仮想環境の設定．

```sh
$ git clone https://github.com/k-onoue/lukasiewicz_2.git
$ cd lukasiewicz_2
$ python3 -m venv logic-env
$ source logic-env/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## 2. 基本的な使用方法

ここでは，このパッケージの基本的な使用方法を紹介します．詳細な手順については，この[Jupyter Notebook](https://github.com/k-onoue/lukasiewicz_2/blob/main/instruction.ipynb)やこの[実験ファイル](https://github.com/k-onoue/lukasiewicz_2/blob/main/experiment_manager/experiment_1.py)を参照してください．

### データの読み込み

```python
import numpy as np
import pandas as pd

df_normal = pd.read_csv(normal_data_path, index_col=0)
X_normal = df_normal.drop(["Target"], axis=1)
y_normal = df_normal["Target"]

# 論理制約の抽出のための離散データ
# 詳細は https://github.com/k-onoue/lukasiewicz_2/blob/main/materials/slide.pdf を参照
df_discrete = pd.read_csv(discrete_data_path, index_col=0)
X_discrete = df_discrete.drop(["Target"], axis=1)
y_discrete = df_discrete["Target"]
```

### 論理制約の抽出

```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.rulefit import RuleFitClassifier, ArrangeRules

# データセットを訓練用とテスト用に分割する
X_train, X_test, y_train, y_test = train_test_split(
    X_discrete.values, y_discrete.values, test_size=0.2, random_state=42
)
feature_names = list(X_discrete.columns)

# RuleFitを使用してルールを取得する
rulefit = RuleFitClassifier(
    rfmode='classify',
    tree_generator=RandomForestClassifier(random_state=42),
    random_state=42,
    exp_rand_tree_size=False
)

rulefit.fit(X_train, y_train, feature_names=feature_names)

# ルールを処理して保存する
rules_df = rulefit.get_rules(exclude_zero_coef=True)
rule_processor = ArrangeRules(
    rules_df,
    feature_names=feature_names,
    conclusion_name="Target"
)
KB_origin = rule_processor.construct_KB()
```

### 入力情報を辞書として設定

```
# 教師ありデータを準備する
L = {}
for col_name in df_discrete.columns:
    df_new = X_normal.iloc[train_idx, :]
    df_new['target'] = df_discrete[col_name].replace(0, -1)
    L[col_name] = df_new

# 教師なしデータを生成する
n_unsupervised = 15
arr_u = np.random.rand(n_unsupervised, X_normal.shape[1])
U = {key: arr_u for key in L.keys()}

# 教師ありデータと教師なしデータを結合する
S = {key: np.vstack([df.drop(['target'], axis=1).values, arr_u]) for key, df in L.items()}

# モデル設定のための入力辞書を準備する
input_dict = {
    'L': L,
    'U': U,
    'S': S,
    'len_j': len(L),
    'len_l': len(train_idx),
    'len_u': n_unsupervised,
    'len_s': len(train_idx) + n_unsupervised,
    'len_h': len(KB_origin),
    'len_i': 2 * n_unsupervised,
    'c1': 15,
    'c2': 15,
    'KB_origin': KB_origin,
    'target_predicate': 'Target',
    # 'kernel_function': "~~logistic regression~~",
}
```

### 予測モデルの訓練

```
from src.setup_problem_primal import SetupPrimal
import cvxpy as cp

# 最適化問題を作成して解く
problem_instance = SetupPrimal(input_dict)
objective_function, constraints = problem_instance.main()
problem = cp.Problem(objective_function, constraints)
result = problem.solve(verbose=True)
```

### 予測

```
p_name = problem_instance.problem_info['target_predicate']
p_trained = problem_instance.problem_info['predicates_dict'][p_name]
y_pred = p_trained(X_test).value
y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)
```

### 予測結果の評価

```
from src.evaluation import evaluate_model

result = evaluate_model(
    y_test,
    y_pred,
    y_pred_interpreted,
    input_for_test, # 参照 https://github.com/k-onoue/lukasiewicz_2/blob/main/instruction.ipynb
    test_idx
)
```


## 3. ディレクトリ構成

```
lukasiewicz_2/
├── data/                               # データセットを格納
│   ├── pima_indian_diabetes/
│   │   ├── diabetes_cleaned_normalized.csv
│   │   ├── diabetes_discretized.csv
├── experiment_manager/                 # 実験の実行ファイル
├── experiment_result/                  # 実験結果を格納
│   ├── tmp/
│   │   ├── version_test/
│   │   │   ├── rules/
│   │   │   │   ├── rules_0.txt
│   │   │   ├── result.json
├── materials/                          # 論文、発表スライドなど
├── notebooks/                          # 実装時の簡易テスト用Jupyterノートブック
├── src/                                # ソースコード
│   ├── association_rule.py             # アソシエーションルールマイニング
│   ├── evaluation.py                   # 予測結果の評価
│   ├── misc.py                         # 各種ユーティリティ
│   ├── objective_function_single_task.py # 双対形式用の目的関数
│   ├── operators.py                    # 論理演算子の変換
│   ├── predicate_single_task.py        # 双対形式用の述語
│   ├── preprocess_fol.py               # 制約論理式の処理
│   ├── rulefit.py                      # RuleFitの修正コード
│   ├── setup_problem_dual_single_task.py # 双対形式用のメインクラス
│   ├── setup_problem_primal.py         # 主形式用のメインクラス
├── instruction.ipynb                   # プログラムの実行手順
├── requirements.txt                    # 依存関係
```


## 4. 質問

質問等あれば，本リポジトリの Issues からお願いします．

## 5. プラットフォーム

<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height=40px>

## 6. 引用

```
尾上 圭介, 小島 諒介:
論理制約を考慮したテーブルデータを対象とした予測モデル構築フレームワーク.
第38回人工知能学会全国大会, 人工知能学会, May., 2024.
https://confit.atlas.jp/guide/event/jsai2024/subject/2M1-OS-11a-02/tables?cryptoId=
```
