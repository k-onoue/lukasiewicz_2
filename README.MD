# Logically Constrained Predictive Models for Tabular Data

This software package provides a framework for extracting logical constraints from tabular data, building predictive models incorporating these constraints, and evaluating the models. It is built on top of CVXPY, a Python-embedded modeling language for convex optimization problems.

__Links__

- [CVXPY GitHub Repository](https://github.com/cvxpy/cvxpy)
- [RuleFit Github Repository](https://github.com/christophM/rulefit)
- [Learning Lukasiewicz Logic Fragments by
Quadratic Programming [2017]](http://ecmlpkdd2017.ijs.si/papers/paperID223.pdf)

__Contents__
- [Logically Constrained Predictive Models for Tabular Data](#logically-constrained-predictive-models-for-tabular-data)
  - [Getting Started](#getting-started)
  - [Basic Usage](#basic-usage)
  - [Directory Structure](#directory-structure)
  - [Questions](#questions)
  - [Supported Platform](#supported-platform)
  - [Citation](#citation)


## Getting Started

To get started, clone the repository and set up the virtual environment:

```sh
$ git clone https://github.com/k-onoue/lukasiewicz_2.git
$ cd lukasiewicz_2
$ python3 -m venv logic-env
$ source logic-env/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```


## Basic Usage


Here we see the basic usage of this package. For more detailed instructions, please refer to [this Jupyter Notebook](https://github.com/k-onoue/lukasiewicz_2/blob/main/instruction.ipynb) or [this experiment file](https://github.com/k-onoue/lukasiewicz_2/blob/main/experiment_manager/experiment_1.py).





## Directory Structure

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


## Questions

Ask questions using the issues section on GitHub.


## Supported Platform

<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height=40px>


## Citation

```
@inproceedings{onoue2014,
  author    = {Keisuke Onoue and Ryosuke Kojima},
  title     = {A framework to construct predictive models with logical constraints for table data},
  booktitle = {Proceedings of the Annual Conference of JSAI},
  year      = {2024},
  month     = {May},
  publisher = {The Japanese Society for Artificial Intelligence},
  url       = {https://www.ai-gakkai.or.jp/jsai2024/en}
}
```






