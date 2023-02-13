# DSC180Flock

This project's model comprises of 3 sub-models (predicting average, amount, and standard deviation) that are used for an algorithm at the end

Targets supported:
* **data** - performs the ETL that extracts data from online, and fills the empty data/raw folder with the raw data
* **features** - performs the data-cleaning and feature engineering for the intermediate data in data/temp folder and final data in data/out folder
* **model** - trains the data on the final data made by the features target, and outputs prediction accuracy metrics and confusion matrix from the validation data
* **clear** - empties the data folders raw, temp, and out
* **all** - complete cycle of ETL, data preparation (cleaning, feature engineering), training, and prediction. Equivalent of "python run.py data features model". 
* **test-data**: all subsequent arguments will be using only the test-data folder, not the data folder.
* **test** - complete cycle but only on the "test data". Equivalent of "python run.py test-data clear feature model"
