# gbtboilerplate

These scripts aim to build a wrapper to train models based on xgboost, lightgbm and catboost.

## Configuration

1. Copy a preprocessed dataset **(no missing values)** in the folder data. 

**IMPORTANT**: The scripts assume the dataset is a csv with a semicolon separator named (dataset.csv).

**IMPORTANT**: The scripts assume there is a column, named ```SET``` which tells if the record is part of the training set ```train```, 
the validation set ```valid``` or the testing set ```test```.

2. Execute the configure bash script:

> sh configure.sh dataset.csv ;

This file will create a config csv file into the config folder containing as row indexes the dataset's columns.
The others columns have to be configured.

3. Edit the config file dataset.csv, just created, in the config folder.

Assume you want to train xgboost, then write under the column ```xgboost```:
- ```index```: if it is the index column
- ```label```: if it is the label
- ```predictor```: if it is a predictor column
and leave it blank otherwise.
Furthermore you may have categorical predictors then you need to mark them with ```1``` under the ```categorical``` column. 

4. Edit the parameters in the **.conf** files.

In particular, in the folder config you find a xgboost subfolder which contains 3 files:
- **xgb_base.conf**: containing the configuration of xgboost
- **xgb_grid.conf**: containing the configuration of a RandomSearchCV grid
- **xgb_hyper.conf**: containing the hyperparameters of the wanted grid

**IMPORTANT**: The scripts assume you want to perform a binary classification.

**IMPORTANT**: The scripts might depend on other python scripts which can be found in this repo.

5. Launch a RandomSearchCV grid:

> python grid.py -d dataset.csv -m xgboost -t xgb

The parameters:
- **d**: refers to the dataset name, in this example dataset.csv
- **m**: refers to the column name of the dataset.csv in the config folder, which we have set in step **3**.
- **t**: refers to the tag name given to the simulation. Notice that the config file names are prepended with the tag name.

This script will output:
- the grid results in a .xlsx file into the folder results
- the best parameters config file which can be found in the subfolder "xgboost" of the folder "config"

6. Train the best model again:

> python gbt.py -d dataset.csv -m xgboost -t xgb