# (Package Name)

This software package provides a framework for extracting logical constraints from tabular data, building predictive models incorporating these constraints, and evaluating the models. It is built on top of cvxpy, a Python-embedded modeling language for convex optimization problems.


各種リンク

https://github.com/cvxpy/cvxpy


**Contents**
- [(Package Name)](#package-name)
  - [Questions](#questions)
  - [Supported platform](#supported-platform)
  - [Citation](#citation)
  - [Getting stated](#getting-stated)
  - [Basic usage](#basic-usage)



## Questions

Ask questions using the issues section.

## Supported platform

[<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height=40px>](https://en.wikipedia.org/wiki/List_of_Linux_distributions)

## Citation

```
@Misc{onoue2024jsai,
  title={A framework to construct predictive models with logical constraints for table data}, 
  author={Keisuke Onoue, Ryosuke Kojima},
  year={2024},
  
}
```

## Getting stated

```
$ git clone https://github.com/k-onoue/lukasiewicz_2.git
$ cd lukasiewicz_2
$ python3 -m venv logic-env
$ source logic-env/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Basic usage

