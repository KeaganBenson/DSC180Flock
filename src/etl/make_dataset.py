#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import pandas as pd

from . import etl_offer_acceptances
from . import etl_zipcode


def main(args):
    #path_folder = "../../"
    path_folder_data = args["path_folder_data"]
    #path_folder_data = os.path.join(path_folder, "data")
    path_folder_data_raw = os.path.join(path_folder_data, "raw")
    etl_zipcode.main(args)
    etl_offer_acceptances.main(args)
    #print(os.listdir(path_folder))
    if "test-data" in str(path_folder_data):
        etl_offer_acceptances.test_data_creator(path_folder_data_raw)
    



# In[ ]:




