# 論理制約付き予測モデル for テーブルデータ

本プログラムは、表形式データから抽出した論理制約を組み込んだ予測モデルの構築から評価までを行うためのフレームワークです。CVXPY（凸最適化ソルバー）を基盤としています。

__リンク__

- [CVXPY GitHub リポジトリ](https://github.com/cvxpy/cvxpy)
- [RuleFit Github Repository](https://github.com/christophM/rulefit)
- [Learning Lukasiewicz Logic Fragments by Quadratic Programming [2017]](http://ecmlpkdd2017.ijs.si/papers/paperID223.pdf)

__目次__
- [論理制約付き予測モデル for テーブルデータ](#論理制約付き予測モデル-for-テーブルデータ)
  - [インストール](#インストール)
  - [ディレクトリ構成](#ディレクトリ構成)
  - [基本的な使用方法](#基本的な使用方法)
  - [質問](#質問)
  - [プラットフォーム](#プラットフォーム)
  - [引用](#引用)


## インストール

リポジトリのクローンと仮想環境の設定．

```sh
$ git clone https://github.com/k-onoue/lukasiewicz_2.git
$ cd lukasiewicz_2
$ python3 -m venv logic-env
$ source logic-env/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```


## ディレクトリ構成

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




## 基本的な使用方法

Coming soon...

## 質問

質問等あれば，本リポジトリの Issues からお願いします．

## プラットフォーム

<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height=40px>

## 引用

```
尾上 圭介, 小島 諒介:
論理制約を考慮したテーブルデータを対象とした予測モデル構築フレームワーク.
第38回人工知能学会全国大会, 人工知能学会, May., 2024.
https://confit.atlas.jp/guide/event/jsai2024/subject/2M1-OS-11a-02/tables?cryptoId=
```