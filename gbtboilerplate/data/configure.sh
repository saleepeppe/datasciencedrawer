config_file="../config/""$1"
echo -e "column;desc;binary;categorical;xgboost;lightgbm;catboost" >> $config_file
sed_param="s/""$2""\|$/;;;;;;\n/g"
head -n 1 "$1" | sed $sed_param | sed '/^$/d' >> $config_file
