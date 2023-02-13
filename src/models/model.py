#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import pandas as pd
import numpy as np
def model(args):
    path_folder_data = args["path_folder_data"]
    file_name_final_df = args["file_name_final_df"]

    path_folder_data_raw = os.path.join(path_folder_data, "raw")
    path_folder_data_out = os.path.join(path_folder_data, "out")

    path_file_oa_orders = os.path.join(path_folder_data_raw, "offer_acceptance_orders.csv")
    path_file_oa_offers = os.path.join(path_folder_data_raw, "offer_acceptance_offers.csv")
    oa_orders = pd.read_csv(path_file_oa_orders)
    oa_offers = pd.read_csv(path_file_oa_offers)

    path_file_final_df = os.path.join(path_folder_data_out, file_name_final_df)
    final_df = pd.read_csv(path_file_final_df)

    oa_orders = oa_orders[oa_orders["TRANSPORT_MODE"]=="FTL"]

    oa = oa_orders.merge(oa_offers,on=["REFERENCE_NUMBER"])

    oa = oa.sort_values(["REFERENCE_NUMBER","CREATED_ON_HQ"])

    final_oa = oa[["REFERENCE_NUMBER","CREATED_ON_HQ","RATE_USD","LOAD_DELIVERED_FROM_OFFER"]]

    final_oa = final_oa.merge(final_df, on=["REFERENCE_NUMBER"])
    print("Final OA column names are:", list(final_oa.columns))

    final_oa["LOG(RATE_USD)"] = np.log1p(final_oa["RATE_USD"])

    # if offer log(rate_usd) is < (PREDICTED_LOG_AVG - PREDICTED_STDEV), then it is a good offer!
    final_oa["OFFER_IS_GOOD"] = (
        final_oa["LOG(RATE_USD)"] < (final_oa["PREDICTED_LOG_AVG"] - final_oa["PREDICTED_STDEV"])
    )

    score = round((final_oa.groupby(["REFERENCE_NUMBER"])["OFFER_IS_GOOD"].sum()==0).mean()* 100,2)
    print("{0} % of the time, we ended up taking no offers at all for an order".format(score))
def main(args):
    model(args)


# In[ ]:




