"""
This file construct a CLI which wraps the main GBT base algorithms to facilitate the trials.
It can be used as boilerplate to customise or directly as python script:
python grid.py -d {csv file containing the data} -m {"xgboost", "lightgbm", "catboost"} -t {string, name of the trial} -n {records' number to use}
"""
import argparse
import os 
import Config as c
import pandas as pd
from pandas_utils import group_categoricals_tail
from sklearn.model_selection import RandomizedSearchCV

def main():
    # Load cli params
    parser = argparse.ArgumentParser(prog="grid")
    parser.add_argument("-d", "--data", dest="data", required=True)
    parser.add_argument("-m", "--model", dest="model", required=True, choices=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("-t", "--tag", dest="tag", required=True)
    parser.add_argument("-n", "--number", dest="number", type=int, default=-1)

    args = parser.parse_args()

    # Load config
    cf = c.Config(args.data, args.model, args.tag)
    grid_params = cf.load_config_file("grid")
    hyper_params = cf.load_config_file("hyper")
    fit_params = {}

    # Load data
    features_to_load = list(cf.features.keys())
    source_file = os.path.join(c.DATA_FOLDER, args.data)
    data = pd.read_csv(source_file, usecols=features_to_load, sep=";", decimal=".", encoding="latin1", keep_default_na = False, na_values = [""])

    # Preprocessing
    group_categoricals_tail(data, cf.classes["categorical"])

    if args.model == "xgboost":
        import xgboost as xgb
        estimator = xgb.XGBClassifier()
        data = pd.get_dummies(data, columns=cf.classes["categorical"]).copy()
    elif args.model == "lightgbm":
        import lightgbm as lgb
        from sklearn.preprocessing import LabelEncoder
        estimator = lgb.LGBMClassifier()
        label_encoding = {}
        for col in cf.classes["categorical"]:
            unique_values = data[col].unique().tolist()
            label_encoding[col] = LabelEncoder()
            label_encoding[col].fit(sorted(unique_values))
            data[col] = label_encoding[col].transform(data[col].values)
        fit_params = {
            "categorical_feature": cf.classes["categorical"]
        }
    elif args.model == "catboost":
        import catboost as cb
        estimator = cb.CatBoostClassifier()
    else: # This code should never end up here
        raise ValueError("Something went wrong: {} is not a feasible model.".format(args.model))
   
    features = [x for x in sorted(data.columns.tolist()) if x not in cf.classes["label"] + cf.classes["index"]]
    if args.model == "lightgbm":
        fit_params["feature_name"] = features
    elif args.model == "catboost":
        fit_params["cat_features"] = [i for i, f in enumerate(features) if f in cf.classes["categorical"]]
        
    # Sampling
    if not args.number == -1 or args.number <= data.shape[0]:
        data = data.sample(args.number)
    d = data.loc[:, features].values

    # Prepare to save
    output = "_".join([args.model, args.tag, "grid"]) + ".xlsx"
    excel_writer = pd.ExcelWriter(os.path.join(c.RESULTS_FOLDER, output), engine="xlsxwriter")

    # Grid
    results = {}

    for label in cf.classes["label"]:
        print("----------------------------", end="\n")
        print(label, end="\n")
        print("----------------------------", end="\n\n")

        y = data.loc[:, label].values

        rand_grid_cv = RandomizedSearchCV(estimator, param_distributions=hyper_params, **grid_params)
        rand_grid_cv.fit(d, y, **fit_params)

        results[label] = rand_grid_cv.best_params_

        results_df = pd.DataFrame(rand_grid_cv.cv_results_)
        results_df.to_excel(excel_writer, sheet_name=label, index=False)

    # Save
    excel_writer.save()
    file_name = "best"
    cf.save_config_file(file_name, results)

if __name__ == "__main__":
    main()