# stat6020-project

This repository contains code for running a group of models on the [Diabetes dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).

## Requirements

* A CSV of the dataset. You can download it from Kaggle
  [here](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). The code in this repo assumes that `diabetes_012.csv` is available in this directory.

* Python 3.9+, with the following packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, and the [`imblearn`](https://imbalanced-learn.org/stable/) package for using the `SMOTE` method.


## Running the code

You can run the code from command-line. For example, to run the AdaBoost estimator

```sh
python adab_estim.py --help
```

```
usage: pyADABOOST [-h] [-o OUT_PATH] [-t TEST_SPLIT] [--using-smote]
                  [--two-class]

options:
  -h, --help            show this help message and exit
  -o OUT_PATH, --out-path OUT_PATH
  -t TEST_SPLIT, --test-split TEST_SPLIT
                        test split
  --using-smote         use smote?
  --two-class           set to true to do 2-class classification
```

* with `-o` you can provide the CSV where the grid-search results will be saved
* with `-t` you can provide a fraction for test split (default is 0.3).

So you can run

```sh
python adab_estim.py --two-class --using-smote -o adab-2c-smote.csv
```

## (Actually) running the code

So the code runs only a small grid search by default -- to run it on a large
grid of hyperparameters, you need to replace the `GLOB_PARAMS["toy"]` with
`GLOB_PARAMS["heavy"]` in the runner function.
