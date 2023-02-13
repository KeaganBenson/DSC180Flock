#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import src.dataclean.build_features as features
import src.etl.create_test_data as create_test_data
import src.etl.make_dataset as etl
import src.models.model as model
import os

def main_etl(args):
    etl.main(args)

def main_dataclean(args):
    #data_clean_first_time = 1
    features.main(args)

def main_model(args):
    model.main(args)

def main_create_test_data(args):
    create_test_data.main(args)

def main_test(args):
    main_clear(args)
    main_create_test_data(args)
    main_dataclean(args)
    main_model(args)    

def main_all(args):
    main_etl(args)
    main_dataclean(args)
    main_model(args)

def main_clear(args):
    path_folder_data = args["path_folder_data"]
    path_folder_data_raw = os.path.join(path_folder_data,"raw")
    path_folder_data_temp = os.path.join(path_folder_data,"temp")
    path_folder_data_out = os.path.join(path_folder_data,"out")
    path_folders_data = [path_folder_data_raw,path_folder_data_temp,path_folder_data_out]
    for path_folder_data in path_folders_data:
        for filename in os.listdir(path_folder_data):
            if filename in [".gitignore.txt", "stub.txt"]:
                print("gitignore")
                continue
            path_file = os.path.join(path_folder_data_raw, filename)
            try:
                os.remove(path_file)
            except:
                pass
        print(os.listdir(path_folder_data_raw))

def main(targets):
    #base_path_folder = "../../"
    base_path_folder = os.getcwd()
    main_path_folder = os.path.join(base_path_folder,"data")
    test_path_folder = os.path.join(base_path_folder,"test","test-data")
    
    args = {
        "path_folder_data": main_path_folder,
        "file_name_temp_avg_stdev": "temp_avg_stdev.csv",
        "file_name_temp_amount": "temp_amount.csv",
        "file_name_final_df": "final_df.csv",
        #"temp_foreign_key_column_name": "REFERENCE",
    }
    for target in targets:
        if target in ["test"]:
            args["path_folder_data"] = test_path_folder
            main_test(args)
        if target in ["all"]:
            main_all(args)
        
        if target in ["data"]:
            main_etl(args)
        if target in ["features"]:
            main_dataclean(args)
        if target in ["model"]:
            main_model(args)
        if target in ["clear"]:
            main_clear(args)
        if target in ["test-data"]:
            args["path_folder_data"] = test_path_folder
        
if __name__ == "__main__":
    targets = sys.argv[1:]
    #targets = ["test"]
    #targets = ["features"]
    main(targets)


# In[ ]:





# In[ ]:




