import argparse
import os 
import Config as c
import numpy as np
import pandas as pd
import pimpmatplotlib as pm
from pandas_utils import group_categoricals_tail
from sklearn.model_selection import RandomizedSearchCV

def main():
    # Load cli params
    parser = argparse.ArgumentParser(prog="gbt")
    parser.add_argument("-d", "--data", dest="data", required=True)
    parser.add_argument("-m", "--model", dest="model", required=True, choices=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("-t", "--tag", dest="tag", required=True)

    args = parser.parse_args()

    # Local config
    PREDICTIONS_FILE = "_".join([args.model, args.tag, "pred.csv"])
    SAVE = True
    pmp = pm.PimpPlot(save=SAVE, folder=os.path.join(c.RESULTS_FOLDER, "plots", args.model))

    # Load config
    cf = c.Config(args.data, args.model, args.tag)
    best_params = cf.load_config_file("best")
    base_params = cf.load_config_file("base")

    # Load data
    features_to_load = list(cf.features.keys()) + ["SET"]
    source_file = os.path.join(c.DATA_FOLDER, args.data)
    data = pd.read_csv(source_file, usecols=features_to_load, sep=";", decimal=".", encoding="latin1", keep_default_na = False, na_values = [""])

    # Preprocessing
    group_categoricals_tail(data, cf.classes["categorical"])

    if args.model == "xgboost":
        import xgboost as xgb
        data = pd.get_dummies(data, columns=cf.classes["categorical"]).copy()
    elif args.model == "lightgbm":
        import lightgbm as lgb
        from sklearn.preprocessing import LabelEncoder
        label_encoding = {}
        for col in cf.classes["categorical"]:
            unique_values = data[col].unique().tolist()
            label_encoding[col] = LabelEncoder()
            label_encoding[col].fit(sorted(unique_values))
            data[col] = label_encoding[col].transform(data[col].values)
    elif args.model == "catboost":
        import catboost as cb
    else: # This code should never end up here
        raise ValueError("Something went wrong: {} is not a feasible model.".format(args.model))

    # Split the dataset
    indexes = {"train": None, "valid": None, "test": None}
    for set_name in indexes.keys():
        indexes[set_name] = np.where(data["SET"] == set_name)[0]

    features = [x for x in sorted(data.columns.tolist()) if x not in cf.classes["label"] + cf.classes["index"] + ["SET"]]
    d = {}
    if args.model == "xgboost":
        for set_name, set_indexes in indexes.items():
            d[set_name] = xgb.DMatrix(data.loc[set_indexes, features])
    elif args.model == "lightgbm":
        for set_name, set_indexes in indexes.items():
            if set_name == "test":
                d[set_name] = data.loc[set_indexes, features].values
            else:
                d[set_name] = lgb.Dataset(data.loc[set_indexes, features], feature_name=features, categorical_feature=cf.classes["categorical"], free_raw_data=False)
    elif args.model == "catboost":
        for set_name, set_indexes in indexes.items():
            d[set_name] = data.loc[set_indexes, features].values
        cat_features = [i for i, f in enumerate(features) if f in cf.classes["categorical"]]

    # Booster
    predictions = {}

    for label in cf.classes["label"]:
        print("----------------------------", end="\n")
        print(label, end="\n")
        print("----------------------------", end="\n\n")

        params = {**best_params[label], **base_params}

        if args.model == "xgboost":
            for set_name, set_indexes in indexes.items():
                d[set_name].set_label(data.loc[set_indexes, label].values)
            y_test = d["test"].get_label()
            bst = xgb.train(params=params, dtrain=d["train"], evals=[(d["valid"], "val")], callbacks=5)
        elif args.model == "lightgbm":
            for set_name, set_indexes in indexes.items():
                if set_name == "test":
                    y_test = data.loc[set_indexes, label].values
                else:
                    d[set_name].set_label(data.loc[set_indexes, label].values)
            bst = lgb.train(params=params, train_set=d["train"], valid_sets=[d["valid"]], early_stopping_rounds=5)
        elif args.model == "catboost":
            y = {}
            for set_name, set_indexes in indexes.items():
                y[set_name] = data.loc[set_indexes, label].values
            y_test = y["test"]
            model = cb.CatBoostClassifier(**params)
            model.fit(d["train"], y["train"], cat_features=cat_features, eval_set=[(d["valid"], y["valid"])])

        pred_label = "{0}_PRED_{1}".format(label, args.tag.upper())
        if args.model == "catboost":
            predictions[pred_label] = model.predict_proba(d["test"])[:, 1]
        else:
            predictions[pred_label] = bst.predict(d["test"])
        title = "_".join([args.tag, label])
        pmp.plot_roc(y_test, predictions[pred_label], title)
        pmp.plot_distributions(y_test, predictions[pred_label], title)
        threshold = pmp.find_threshold_max_f1(y_test, predictions[pred_label], title, N = 100)
        binary_predictions = np.where(predictions[pred_label] >= threshold, 1, 0)
        pmp.plot_confusion_matrix(y_test, binary_predictions, [0, 1], title)
    
        if SAVE:
            if args.model == "catboost":
                model.save_model(os.path.join(c.MODELS_FOLDER, title + ".model"))
            else:
                bst.save_model(os.path.join(c.MODELS_FOLDER, title + ".model"))
    
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(os.path.join(c.RESULTS_FOLDER, PREDICTIONS_FILE), sep=";", index=False)

if __name__ == "__main__":
    main()