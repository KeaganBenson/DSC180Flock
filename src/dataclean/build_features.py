#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (5,3)
import tqdm
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import pylab as pl
from matplotlib import collections  as mc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

path_folder = "../../"
import sys
sys.path.insert(0, path_folder+"/src/"#+features/"
                )
import util




def build_features(args):
    path_folder = args["path_folder"]
    path_folder_data = os.path.join(path_folder, "data")
    path_folder_data_temp = os.path.join(path_folder_data, "temp")
    path_folder_data_final = os.path.join(path_folder_data, "out")


    file_name_temp_avg_stdev = args["file_name_temp_avg_stdev"]
    file_name_temp_amount = args["file_name_temp_amount"]
    file_name_final_df = args["file_name_final_df"]
    temp_foreign_key_column_name = args["temp_foreign_key_column_name"]
    
    # temp_foreign_key_column_name is basically "REFERENCE"
    path_file_temp_avg_stdev = os.path.join(path_folder_data_temp,file_name_temp_avg_stdev)
    path_file_temp_amount = os.path.join(path_folder_data_temp, file_name_temp_amount)

    temp_avg_stdev = pd.read_csv(path_file_temp_avg_stdev)
    temp_amount = pd.read_csv(path_file_temp_amount)

    final_df = temp_amount.merge(temp_avg_stdev, on=[temp_foreign_key_column_name])
    path_folder_data_final = os.path.join(path_folder_data, "out")
    final_df.to_csv(path_folder_data_final, "file_name_final_df")
    return final_df

def main(args):
    final_df = build_features(args)

