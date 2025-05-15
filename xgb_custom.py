import os
import json
import argparse

#
import sklearn
import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    make_scorer,
    confusion_matrix,
)

#
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

METHOD = "XGBOOST-CUSTOM"


def asymm2_obj(labels: np.ndarray, predt: np.ndarray):
    alphas = np.array([1.0, 2.5, 1.25])
    betas = np.array([1.0, 3.5, 1.25])
    #
    rows = labels.shape[0]
    classes = 1
    #
    alphas = alphas[0]
    betas = betas[1]
    #
    grad = np.zeros((rows, classes), dtype=np.float64)
    hess = np.zeros((rows, classes), dtype=np.float64)
    eps = 1e-6
    #
    target = np.int32(labels)
    #
    p = 1 / (1 + np.exp(predt))
    grad = (target == 0) * (2 * p * alphas) + (target == 1) * 2 * (1 - p) * betas
    hess = np.maximum(2 * p * (1 - p), eps)
    return grad, hess


def asymm3_obj(labels: np.ndarray, predt: np.ndarray):
    alphas = np.array([1.0, 2.5, 1.25])
    betas = np.array([1.0, 3.5, 1.25])
    #
    rows = labels.shape[0]
    classes = predt.shape[1]
    inds = np.arange(rows)
    #
    alphas = alphas[:classes]
    betas = betas[:classes]
    #
    grad = np.zeros((rows, classes), dtype=np.float64)
    hess = np.zeros((rows, classes), dtype=np.float64)
    eps = 1e-6
    #
    target = np.int32(labels)
    t = betas[target]
    #
    p = softmax(predt, axis=1)
    pp = p[inds, target]
    grad = 2 * p * alphas
    grad[inds, target] = 2 * (1 - pp) * t
    hess[inds, target] = np.maximum(2 * pp * (1 - pp), eps)
    return grad, hess


GLOB_PARAMS = {
    "toy": {
        "raw": {
            "clf__n_estimators": (3, 4),
            "clf__learning_rate": (0.5,),
            "clf__tree_method": ("hist",),
            "clf__n_jobs": (None,),
            "clf__objective": (asymm3_obj,),
        },
        "smote": {
            "samp__k_neighbors": (2,),
            "clf__n_estimators": (3, 4),
            "clf__learning_rate": (0.5,),
            "clf__tree_method": ("hist",),
            "clf__n_jobs": (None,),
            "clf__objective": (asymm3_obj,),
        },
    },
    "heavy": {
        "raw": {
            "clf__n_estimators": tuple(np.arange(2, 51, 3)),
            "clf__learning_rate": tuple(np.arange(0.1, 2, 0.3)),
            "clf__tree_method": ("hist",),
            "clf__n_jobs": (None,),
            "clf__objective": (asymm3_obj,),
        },
        "smote": {
            "samp__k_neighbors": (2, 3, 5, 10),
            "clf__n_estimators": tuple(np.arange(2, 51, 3)),
            "clf__learning_rate": tuple(np.arange(0.1, 2, 0.3)),
            "clf__tree_method": ("hist",),
            "clf__n_jobs": (None,),
            "clf__objective": (asymm3_obj,),
        },
    },
}


def preproc(data, two_class, test_size):
    for x in data.columns:
        uvc = data[x].unique()
        if x == "Diabetes_012" or len(uvc) == 2:
            data[x] = np.int32(data[x])

    X = data.iloc[:, 1:]
    if two_class:
        Y = np.array(data.iloc[:, 0] != 0, dtype=np.int32)
    else:
        Y = np.array(data.iloc[:, 0])

    few_unique_cols = [col for col in X.columns if X[col].nunique() <= 3]
    last_two_cols = X.columns[-2:].tolist()
    categorical_cols = list(set(few_unique_cols + last_two_cols))
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    X_train0, X_test0, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=9983
    )

    psor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train = psor.fit_transform(X_train0)
    X_train = pd.DataFrame(X_train, columns=psor.get_feature_names_out())

    X_test = psor.transform(X_test0)
    X_test = pd.DataFrame(X_test, columns=psor.get_feature_names_out())

    return X_train, Y_train, X_test, Y_test


def custom_3scorer(y_true, y_pred):
    # repo = classification_report(y_true, y_pred, output_dict=True)
    # return (
    #    0.3 * repo["0"]["f1-score"] + 5 * repo["1"]["f1-score"] + repo["2"]["f1-score"]
    # )
    s1 = f1_score(y_true, y_pred, labels=[0], average="micro")
    s2 = f1_score(y_true, y_pred, labels=[1], average="micro")
    s3 = f1_score(y_true, y_pred, labels=[2], average="micro")
    return 0.3 * s1 + 5 * s2 + s3


def runner(
    data,
    out_path,
    test_split=0.2,
    two_class=False,
    using_smote=False,
):
    # splits
    X_train, Y_train, X_test, Y_test = preproc(data, two_class, test_split)

    # classifier
    clf = xgb.XGBClassifier(n_estimators=5, learning_rate=0.7)

    # k-fold
    rkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1729)

    # smote
    if using_smote:
        ros = SMOTE(sampling_strategy="not majority", random_state=31415)
        model = Pipeline([("samp", ros), ("clf", clf)])
        params = GLOB_PARAMS["toy"]["smote"]
    else:
        model = Pipeline([("clf", clf)])
        params = GLOB_PARAMS["toy"]["raw"]

    # two-class or three-class?
    if two_class:
        scorer = make_scorer(f1_score, pos_label=1)
        params["clf__objective"] = (asymm2_obj,)
    else:
        scorer = make_scorer(custom_3scorer)

    # setup result
    gs = GridSearchCV(
        model, param_grid=params, cv=rkf, scoring=scorer, error_score="raise", n_jobs=1
    )

    gs.fit(X_train, Y_train)

    rdf = gs.cv_results_
    rdf = pd.DataFrame(rdf)
    print(rdf)

    best_model = gs.best_estimator_
    Y_pred = best_model.predict(X_test)

    rdf.to_csv(out_path, header=True, index=False)
    rdf["method"] = METHOD

    pred_path = os.path.splitext(out_path)[0] + ".json"
    report = classification_report(Y_test, Y_pred, output_dict=True)
    cmat = confusion_matrix(Y_test, Y_pred)
    report["confusion_matrix"] = cmat.tolist()
    report["best_params"] = gs.best_params_
    report["best_params"].pop("clf__objective")

    print(f"Best {METHOD} model evaluation")
    print(classification_report(Y_test, Y_pred))
    print("Confusion Matrix")
    print(cmat)

    with open(pred_path, "w") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(f"py{METHOD}")
    parser.add_argument("-o", "--out-path", default="", type=str)
    parser.add_argument(
        "-t", "--test-split", default=0.3, type=float, help="test split"
    )
    parser.add_argument("--using-smote", action="store_true", help="use smote?")
    parser.add_argument(
        "--two-class",
        action="store_true",
        default=False,
        help="set to true to do 2-class classification",
    )

    d = parser.parse_args()
    if d.out_path == "":
        c = 2 if d.two_class else 3
        s = "smote" if d.using_smote else "raw"
        d.out_path = f"./{METHOD}-cls{c}-{s}.csv"
    print(d)
    df = pd.read_csv("./diabetes_012.csv", header=0)
    runner(
        df,
        #
        out_path=d.out_path,
        test_split=d.test_split,
        using_smote=d.using_smote,
        two_class=d.two_class,
    )


if __name__ == "__main__":
    main()
