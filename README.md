# DSC180Flock

This project's model comprises of 3 sub-models (predicting average, amount, and standard deviation) that are used for an algorithm at the end
On Command Prompt,for the first time, run the following to clone the repo for the first time:
```
git clone https://github.com/KeaganBenson/DSC180Flock.git
```
Then open Anaconda Prompt, for the first time, enter the new folder, and run the following to create a new conda environment with the requirements.txt. 
```
cd dsc180flock
conda create --name flock_env --file requirements.txt
conda activate flock_env
python run.py all
```
While python run.py all is being ran, plots and maps may open up on seperate windows during the execution, and the execution pauses until those windows are closed. After the execution is complete, observe the metrics printed in the console and close the anaconda
Now that the repo is cloned locally and the environment is created, anytime you want to run the model again, you open anaconda prompt and run:
```
cd dsc180flock
conda activate flock_env
python run.py all
```


Targets supported:
* **data** - performs the ETL that extracts data from online, and fills the empty data/raw folder with the raw data
* **features** - performs the data-cleaning and feature engineering for the intermediate data in data/temp folder and final data in data/out folder
* **model** - trains the data on the final data made by the features target, and outputs prediction accuracy metrics and confusion matrix from the validation data
* **clear** - empties the data folders raw, temp, and out
* **all** - complete cycle of ETL, data preparation (cleaning, feature engineering), training, and prediction. Equivalent of "python run.py clear data features model". 
* **test-data**: all subsequent arguments will be using only the test-data folder, not the data folder.
* **test** - complete cycle but only on the "test data". Equivalent of "python run.py test-data clear feature model"



