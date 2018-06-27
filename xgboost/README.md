# xgboost

## Callbacks
We have implemented an early stopping xgboost callback based and on behaviour of the stopping metric over the validation set.

If the stopping metric performances on validation set are less than a tolerance value, stop the training.
The performances of the metric are provided by:

![f1]

where ![f2] is the metric and the subscripts identify the training rounds.

[f1]: https://chart.apis.google.com/chart?cht=tx&chl=\frac{\left%20|{m}_{best}-{m}_{secondbest}\right%20|}{{m}_{last}}
[f2]: https://chart.apis.google.com/chart?cht=tx&chl=m