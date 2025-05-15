import os
import json
import argparse

#
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
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

METHOD = "LSVC"

GLOB_PARAMS = {
    "toy": {
        "raw": {"clf__C": (0.2, 1.0)},
        "smote": {
            "samp__k_neighbors": (2,),
            "clf__C": (0.2, 1.0),
        },
    },
    "heavy": {
        "raw": {
            "clf__C": tuple(np.arange(0.01, 5, 0.25).tolist() + [10, 25, 50, 100]),
        },
        "smote": {
            "samp__k_neighbors": (2, 3, 5, 10),
            "clf__C": tuple(np.arange(0.01, 5, 0.25).tolist() + [10, 25, 50, 100]),
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
    clf = LinearSVC(C=1.0)

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
    else:
        scorer = make_scorer(custom_3scorer)

    # setup result
    gs = GridSearchCV(
        model, param_grid=params, cv=rkf, scoring=scorer, error_score="raise", n_jobs=-1
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
