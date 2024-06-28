# Logically Constrained Predictive Models for Tabular Data

This software package provides a framework for extracting logical constraints from tabular data, building predictive models incorporating these constraints, and evaluating the models. It is built on top of CVXPY, a Python-embedded modeling language for convex optimization problems.  

__Links__

- [CVXPY GitHub Repository](https://github.com/cvxpy/cvxpy)
- [RuleFit Github Repository](https://github.com/christophM/rulefit)
- [Learning Lukasiewicz Logic Fragments by
Quadratic Programming [2017]](http://ecmlpkdd2017.ijs.si/papers/paperID223.pdf)

__Contents__
- [Logically Constrained Predictive Models for Tabular Data](#logically-constrained-predictive-models-for-tabular-data)
  - [1. Getting Started](#1-getting-started)
  - [2. Basic Usage](#2-basic-usage)
    - [Load data](#load-data)
    - [Extract logical constraints from tabular data](#extract-logical-constraints-from-tabular-data)
    - [Configure input information as a dictionary](#configure-input-information-as-a-dictionary)
    - [Train a predictive model](#train-a-predictive-model)
    - [Making predictions](#making-predictions)
    - [Evaluate predictions](#evaluate-predictions)
  - [3. Directory Structure](#3-directory-structure)
  - [4. Questions](#4-questions)
  - [5. Supported Platform](#5-supported-platform)
  - [6. Citation](#6-citation)


## 1. Getting Started

To get started, clone the repository and set up the virtual environment:

```sh
$ git clone https://github.com/k-onoue/lukasiewicz_2.git
$ cd lukasiewicz_2
$ python3 -m venv logic-env
$ source logic-env/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```


## 2. Basic Usage

Here we see the basic usage of this package. For more detailed instructions, please refer to [this Jupyter Notebook](https://github.com/k-onoue/lukasiewicz_2/blob/main/instruction.ipynb) or [this experiment file](https://github.com/k-onoue/lukasiewicz_2/blob/main/experiment_manager/experiment_1.py).

### Load data

```
import numpy as np
import pandas as pd

df_normal = pd.read_csv(normal_data_path, index_col=0)
X_normal = df_normal.drop(["Target"], axis=1)
y_normal = df_normal["Target"]

# Discretized data for the extraction of logical constraints
# See https://github.com/k-onoue/lukasiewicz_2/blob/main/materials/slide.pdf
df_discrete = pd.read_csv(discrete_data_path, index_col=0)
X_discrete = df_discrete.drop(["Target"], axis=1)
y_discrete = df_discrete["Target"]
```

### Extract logical constraints from tabular data

```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.rulefit import RuleFitClassifier, ArrangeRules

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_discrete.values, y_discrete.values, test_size=0.2, random_state=42
)
feature_names = list(X_discrete.columns)

# Acquire rules using RuleFit
rulefit = RuleFitClassifier(
    rfmode='classify',
    tree_generator=RandomForestClassifier(random_state=42),
    random_state=42,
    exp_rand_tree_size=False
)

rulefit.fit(X_train, y_train, feature_names=feature_names)

# Process and save rules
rules_df = rulefit.get_rules(exclude_zero_coef=True)
rule_processor = ArrangeRules(
    rules_df,
    feature_names=feature_names,
    conclusion_name="Target"
)
KB_origin = rule_processor.construct_KB()
```

### Configure input information as a dictionary

```
# Prepare training data
L = {}
for col_name in df_discrete.columns:
    df_new = X_normal.iloc[train_idx, :]
    df_new['target'] = df_discrete[col_name].replace(0, -1)
    L[col_name] = df_new

# Generate unsupervised data
n_unsupervised = 15
arr_u = np.random.rand(n_unsupervised, X_normal.shape[1])
U = {key: arr_u for key in L.keys()}

# Combine training and unsupervised data
S = {key: np.vstack([df.drop(['target'], axis=1).values, arr_u]) for key, df in L.items()}

# Prepare input dictionary for model setup
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

### Train a predictive model

```
# Prepare training data
L = {}
for col_name in df_discrete.columns:
    df_new = X_normal.iloc[train_idx, :]
    df_new['target'] = df_discrete[col_name].replace(0, -1)
    L[col_name] = df_new

# Generate unsupervised data
n_unsupervised = 15
arr_u = np.random.rand(n_unsupervised, X_normal.shape[1])
U = {key: arr_u for key in L.keys()}

# Combine training and unsupervised data
S = {key: np.vstack([df.drop(['target'], axis=1).values, arr_u]) for key, df in L.items()}

# Prepare input dictionary for model setup
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

### Making predictions

```
p_name = problem_instance.problem_info['target_predicate']
p_trained = problem_instance.problem_info['predicates_dict'][p_name]
y_pred = p_trained(X_test).value
y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)
```

### Evaluate predictions

```
result = evaluate_model(
    y_test,
    y_pred,
    y_pred_interpreted,
    input_for_test, # refer to https://github.com/k-onoue/lukasiewicz_2/blob/main/instruction.ipynb
    test_idx
)
```



## 3. Directory Structure

```
## Directory Structure

lukasiewicz_2/
├── data/                               # Contains datasets
│   ├── pima_indian_diabetes/
│   │   ├── diabetes_cleaned_normalized.csv
│   │   ├── diabetes_discretized.csv
├── experiment_manager/                 # Executable files for running experiments
├── experiment_result/                  # Stores experiment results
│   ├── tmp/
│   │   ├── version_test/
│   │   │   ├── rules/
│   │   │   │   ├── rules_0.txt
│   │   │   ├── result.json
├── materials/                          # Contains papers, presentation slides, etc.
├── notebooks/                          # Jupyter notebooks for simple tests during implementation
├── src/                                # Source code
│   ├── association_rule.py             # Association Rule Mining
│   ├── evaluation.py                   # Evaluation of prediction results
│   ├── misc.py                         # Utility functions
│   ├── objective_function_single_task.py # Objective function for dual form
│   ├── operators.py                    # Conversion of logical operators
│   ├── predicate_single_task.py        # Predicates for dual form
│   ├── preprocess_fol.py               # Processing of constraint logical expressions
│   ├── rulefit.py                      # Modified code for Rulefit implemented by christophM
│   ├── setup_problem_dual_single_task.py # Main class for dual form
│   ├── setup_problem_primal.py         # Main class for primal form
├── instruction.ipynb                   # Instructions for running the program
├── requirements.txt                    # Dependencies
```


## 4. Questions

Ask questions using the issues section on GitHub.


## 5. Supported Platform

<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height=40px>


## 6. Citation

```
@inproceedings{onoue2024,
  author    = {Keisuke Onoue and Ryosuke Kojima},
  title     = {A framework to construct predictive models with logical constraints for table data},
  booktitle = {Proceedings of the Annual Conference of JSAI},
  year      = {2024},
  month     = {May},
  publisher = {The Japanese Society for Artificial Intelligence},
  url       = {https://www.ai-gakkai.or.jp/jsai2024/en}
}
```






