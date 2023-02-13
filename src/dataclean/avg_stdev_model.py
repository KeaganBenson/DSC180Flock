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

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

#path_folder = os.getcwd()
def dataclean(path_folder = "../../", file_name="temp_avg_stdev.csv"):
    file_name_temp_avg_stdev = file_name
    path_folder_data = os.path.join(path_folder, "data")
    path_folder_data_raw = os.path.join(path_folder_data,"raw")
    path_folder_data_temp = os.path.join(path_folder_data,"temp")
    path_folder_data_final = os.path.join(path_folder_data,"final")

    file_name_oa_offers = "offer_acceptance_offers.csv"
    path_file_oa_offers = os.path.join(path_folder_data_raw, file_name_oa_offers)
    oa_offers = pd.read_csv(path_file_oa_offers)
    print("Loaded offers df")
    print(oa_offers.shape)

    file_name_oa_orders = "offer_acceptance_orders.csv"
    path_file_oa_orders = os.path.join(path_folder_data_raw, file_name_oa_orders)
    oa_orders = pd.read_csv(path_file_oa_orders)
    print("Loaded orders df")
    print(oa_orders.shape)

    file_name_zipcode_coordinates = "zipcode_coordinates.csv"
    path_file_zipcode_coordinates = os.path.join(path_folder_data_raw, file_name_zipcode_coordinates)
    zipcode_coordinates = pd.read_csv(path_file_zipcode_coordinates)
    print("Loaded zipcodes df")
    print(zipcode_coordinates.shape)

    ## Data Cleaning

    #weighing_by_lead_time = False
    #filtering_ftl = False
    weighing_by_lead_time = 0
    filtering_ftl = 0



    # data cleaning offers
    # In this part, I'm just adding columns to the oa offers table

    oa_offers_boolean_column_names = ['SELF_SERVE', 'IS_OFFER_APPROVED',
               'AUTOMATICALLY_APPROVED', 'MANUALLY_APPROVED', 'WAS_EVER_UNCOVERED',
               'COVERING_OFFER', 'LOAD_DELIVERED_FROM_OFFER', 'RECOMMENDED_LOAD', 'VALID']
    oa_offers_date_column_names = ["CREATED_ON_HQ"]
    oa_offers_categorical_column_names = ["OFFER_TYPE"]
    oa_offers_loggable_numerical_column_names = ["RATE_USD"]

    oa_offers = util.add_cleaned_reference_numbers_column(oa_offers)

    oa_offers = util.add_num_reference_numbers_column(oa_offers)

    oa_offers = util.dataclean_pool_reference_numbers_discrepancy(oa_offers)

    offer_type_is_pooled_column_name = "OFFER_TYPE_IS_POOLED"
    oa_offers = util.add_offer_type_is_pooled_column(oa_offers,column_name=offer_type_is_pooled_column_name)
    oa_offers_boolean_column_names += [offer_type_is_pooled_column_name]

    for column_name in oa_offers_date_column_names:
        oa_offers = util.convert_pd_datetime_column(oa_offers, column_name)
    for column_name in oa_offers_boolean_column_names:
        oa_offers = util.convert_boolean_to_num_column(oa_offers, column_name)
    for column_name in oa_offers_loggable_numerical_column_names:
        oa_offers = util.add_log_column_to_df(oa_offers, column_name)

    zipcode_coordinates["X_COORD"] /= 1.0e6
    zipcode_coordinates["Y_COORD"] /= 1.0e6
    zipcode_coordinates["3DIGIT_ZIP"] = zipcode_coordinates["3DIGIT_ZIP"].astype(int).astype(str).str.zfill(3)

    oa_orders = (
        oa_orders
        .merge(zipcode_coordinates, left_on=["DESTINATION_3DIGIT_ZIP"], right_on=["3DIGIT_ZIP"])
        .rename(columns={"X_COORD":"X_COORD_DEST","Y_COORD":"Y_COORD_DEST"})
        .drop(columns=["3DIGIT_ZIP"])
    )

    oa_orders = (
        oa_orders
        .merge(zipcode_coordinates, left_on=["ORIGIN_3DIGIT_ZIP"], right_on=["3DIGIT_ZIP"])
        .rename(columns={"X_COORD":"X_COORD_ORIG","Y_COORD":"Y_COORD_ORIG"})
        .drop(columns=["3DIGIT_ZIP"])
    )



    # data cleaning orders
    # In this part, I'm just adding columns to the oa orders table

    oa_orders_boolean_column_names = ['FD_ENABLED', 'EXCLUSIVE_USE_REQUESTED','HAZARDOUS', 
                                      'REEFER_ALLOWED', 'STRAIGHT_TRUCK_ALLOWED','LOAD_TO_RIDE_REQUESTED']
    oa_orders_loggable_numerical_column_names = ["ESTIMATED_COST_AT_ORDER","APPROXIMATE_DRIVING_ROUTE_MILEAGE",
                                                 "PALLETIZED_LINEAR_FEET","LOAD_BAR_COUNT"]
    oa_orders_date_column_names = ["ORDER_DATETIME_PST","PICKUP_DEADLINE_PST"]
    oa_orders_categorical_column_names = ["DELIVERY_TIME_CONSTRAINT","TRANSPORT_MODE"]

    if filtering_ftl == True:
        oa_orders = util.dataclean_filter_ftl(oa_orders)
        # drop transport mode column from categorical column names to prevent a single-category one-hot-encoding
        oa_orders_categorical_column_names.remove("TRANSPORT_MODE")
    else:
        pass

    # Matt said that any duplicate references is likely an error
    oa_orders = util.dataclean_ftl_duplicate_references(oa_orders)

    oa_orders = util.add_cleaned_reference_numbers_column(oa_orders)

    distance_column_name = "APPROXIMATE_DRIVING_ROUTE_MILEAGE"
    oa_orders = util.add_distance_column(oa_orders,column_name=distance_column_name)
    oa_orders_loggable_numerical_column_names += [distance_column_name]

    time_between_order_and_deadline_column_name = "TIME_BETWEEN_ORDER_AND_DEADLINE"
    oa_orders = util.add_time_between_order_and_deadline_column(oa_orders,column_name=time_between_order_and_deadline_column_name)


    #month_column_names = ["MONTH_"+str(x) for x in range(1,13)]
    #oa_orders = add_ohe_month_columns(oa_orders, column_names=month_column_names)
    #oa_orders_boolean_column_names += month_column_names

    #weekday_column_names = ["DAY_"+str(x) for x in range(0,7)]
    #oa_orders = add_ohe_weekday_columns(oa_orders, column_names=weekday_column_names)
    #oa_orders_boolean_column_names += weekday_column_names

    month_column_name = "MONTH"
    oa_orders = util.add_month_column(oa_orders, column_name=month_column_name)
    oa_orders_categorical_column_names += [month_column_name]

    weekday_column_name = "DAY"
    oa_orders = util.add_weekday_column(oa_orders, column_name=weekday_column_name)
    oa_orders_categorical_column_names += [weekday_column_name]

    for column_name in oa_orders_categorical_column_names:
        ohe_columns = util.temp_build_ohe(oa_orders[column_name],column_name=column_name)
        ohe_column_names = list(ohe_columns.columns)
        oa_orders_boolean_column_names += ohe_column_names
        oa_orders = util.add_ohe_columns(oa_orders, ohe_columns)
        oa_orders.drop(columns=[column_name],inplace=True)

    for column_name in oa_orders_date_column_names:
        oa_orders = util.convert_pd_datetime_column(oa_orders, column_name)
    for column_name in oa_orders_boolean_column_names:
        oa_orders = util.convert_boolean_to_num_column(oa_orders, column_name)
    for column_name in oa_orders_loggable_numerical_column_names:
        oa_orders = util.add_log_column_to_df(oa_orders, column_name)

    #oa_orders = oa_orders.drop(columns=["REFERENCE_NUMBER"])




    # joining the oa_offers and oa_orders into a new table called oa
    # oa will have 2 new columns "SD_LOG(RATE_USD)" and "ORDER_OFFER_AMOUNT" 
    # which are the standard deviation of the logged rate_usd, and the actual number of offers

    log_rate_usd_column_name = "LOG(RATE_USD)"
    count_reference_numbers_column_name = "ORDER_OFFER_AMOUNT"
    sd_log_rate_usd_column_name = "SD_LOG(RATE_USD)"
    # adding a duplicate of the rates column to apply standard deviaton later on during the group by
    oa_offers[sd_log_rate_usd_column_name] = oa_offers[log_rate_usd_column_name]
    # adding a duplicate of the rates column to apply count later on during the group by
    oa_offers[count_reference_numbers_column_name] = oa_offers["REFERENCE_NUMBER"]


    # Recall that in oa_offers, the column "REFERENCE_NUMBERS" are lists of reference number strings, with at max 13 items
    assert (type(oa_offers["REFERENCE_NUMBERS"].values[0]) == list)
    #print("oa_offers[REFERENCE_NUMBERS] can have up to # items:", 
    #      (oa_offers["REFERENCE_NUMBERS"].apply(len)).max())
    #print("oa_offers[REFERENCE_NUMBERS].values[0] is", (oa_offers["REFERENCE_NUMBERS"].values[0]))

    # So we should apply the pandas explode function to the REFERENCE_NUMBERS column in oa_offers
    oa_offers_exploded = oa_offers.explode(["REFERENCE_NUMBERS"])
    # Now, oa_offers has more rows, and reference numbers are now strings
    assert (type(oa_offers_exploded["REFERENCE_NUMBERS"].values[0]) == str)
    #print("oa_offers_exploded[REFERENCE_NUMBERS].values[0] is", (oa_offers_exploded["REFERENCE_NUMBERS"].values[0]))
    #print("oa_offers_exploded[REFERENCE_NUMBERS].values[1] is", (oa_offers_exploded["REFERENCE_NUMBERS"].values[1]))
    assert ((oa_offers["REFERENCE_NUMBERS"].apply(len)).sum() == oa_offers_exploded.shape[0])

    # now, the newly exploded oa_offers must be groupby'd on REFERENCE_NUMBERS to be in the same joinable level as oa orders
    aggdict = dict()
    aggdict[log_rate_usd_column_name] = np.mean 
    aggdict[sd_log_rate_usd_column_name] = np.std # standard deviation
    aggdict[count_reference_numbers_column_name] = "count"
    oa_offers_groupby = oa_offers_exploded.groupby(["REFERENCE_NUMBERS"],as_index=False).agg(aggdict)

    #print("")
    # Recall that in oa_orders, the column "REFERENCE_NUMBERS" are also lists of reference number strings,
    # but they only have 1 item
    assert (type(oa_orders["REFERENCE_NUMBERS"].values[0]) == list)
    #print("oa_orders[REFERENCE_NUMBERS].values[0] is", (oa_orders["REFERENCE_NUMBERS"].values[0]))
    assert oa_orders["REFERENCE_NUMBERS"].apply(len).max() == 1
    assert oa_orders["REFERENCE_NUMBERS"].apply(len).min() == 1
    assert ((oa_orders["REFERENCE_NUMBERS"].apply(len)).sum() == oa_orders.shape[0])

    # Since it's just 1 item, exploding REFERENCE_NUMBERS is no different 
    # than just doing oa_orders["REFERENCE_NUMBERS"] = oa_orders["REFERENCE_NUMBERS"].apply(lambda x: x[0])
    oa_orders_exploded = oa_orders.explode(["REFERENCE_NUMBERS"])
    assert ((oa_orders.shape[0] == oa_orders_exploded.shape[0]))
    assert (type(oa_orders_exploded["REFERENCE_NUMBERS"].values[0]) == str)
    #print("oa_orders_exploded[REFERENCE_NUMBERS].values[0] is", oa_orders_exploded["REFERENCE_NUMBERS"].values[0])

    # finally merging oa_offers and oa_orders
    foreign_key_column_name = "REFERENCE_NUMBERS"

    temp_oa_orders = oa_orders_exploded
    temp_oa_offers = oa_offers_groupby
    oa = temp_oa_orders.merge(temp_oa_offers, on=[foreign_key_column_name])

    oa_boolean_column_names = []
    oa_loggable_numerical_column_names = []
    oa_numerical_column_names = []
    oa_categorical_column_names = []


    distance_over_order_time_column_name = "DISTANCE_OVER_ORDER_TIME"
    oa = util.add_distance_over_order_time_column(oa,column_name=distance_over_order_time_column_name)
    oa_numerical_column_names += [distance_over_order_time_column_name]

    metro_cluster = util.temp_build_metro_cluster(oa)
    oa = util.add_metro_cluster_columns(oa, metro_cluster)

    lead_time_column_name = "LEAD_TIME"
    weight_column_name = None
    if weighing_by_lead_time == True:
        weight_column_name = lead_time_column_name
        oa = util.add_lead_time_column(oa, oa_offers_exploded, column_name=lead_time_column_name)
    else:
        pass

    #offer_order_amount_column_name = "ORDER_OFFER_AMOUNT"
    #oa = add_offer_order_amount_column(oa, column_name=offer_order_amount_column_name)
    oa_loggable_numerical_column_names += [count_reference_numbers_column_name]

    for column_name in oa_loggable_numerical_column_names:
        oa = util.add_log_column_to_df(oa, column_name)

    oa_collapsed_orders = oa

    #is_singleton = (oa_collapsed_orders[count_reference_numbers_column_name]==1).mean()
    #is_null = oa_collapsed_orders[sd_log_rate_usd_column_name].isnull().mean()
    #assert is_singleton == is_null
    oa_collapsed_orders.replace([np.inf, -np.inf], np.nan, inplace=True)
    oa_collapsed_orders = oa_collapsed_orders.fillna(0)

    oa_collapsed_orders = oa_collapsed_orders.drop(columns=[
        'ORDER_DATETIME_PST', 
        'PICKUP_DEADLINE_PST',
        'ORIGIN_3DIGIT_ZIP', 
        'DESTINATION_3DIGIT_ZIP',
        'LOG(ESTIMATED_COST_AT_ORDER)',
        'ESTIMATED_COST_AT_ORDER',
        'REFERENCE_NUMBERS',
        'REFERENCE_NUMBER'
    ])

    ## Now building the models for predicting the Avg and StDev

    # avg model
    target_column_name="LOG(RATE_USD)"
    input_df = oa_collapsed_orders.drop(columns=[
        "SD_LOG(RATE_USD)",
        "ORDER_OFFER_AMOUNT",
        "LOG(ORDER_OFFER_AMOUNT)"
    ])

    avg_model_builder = util.Temp_Order_Regression_Model_Builder(
        df=input_df,
        target_column_name=target_column_name,
        weight_column_name=weight_column_name
    )
    avg_model_builder.transform_y(inplace=True,update_df=True)
    avg_model_builder.transform_X(inplace=True)
    avg_model_builder.set_split()
    avg_model_builder.train_model(LinearRegression())
    predictions = avg_model_builder.test_model()
    avg_model_builder.eval_model(predictions)
    #plt.scatter(predictions, avg_model_builder.y_test,s=5,alpha=0.2)

    # num model

    target_column_name="ORDER_OFFER_AMOUNT"
    input_df = oa_collapsed_orders.drop(columns=[
        "SD_LOG(RATE_USD)",
        "LOG(RATE_USD)",
        "LOG(ORDER_OFFER_AMOUNT)"
    ])
    predicted_avg_column = avg_model_builder.model.predict(
        input_df[avg_model_builder.top_n_correlated_selected_column_names])
    input_df["PRED_LOG(RATE_USD)"] = predicted_avg_column
    num_model_builder = util.Temp_Order_Classification_Model_Builder(
        df=input_df,
        target_column_name=target_column_name,
        weight_column_name=weight_column_name
    )
    num_model_builder.transform_y(inplace=True)
    num_model_builder.transform_X(inplace=True)
    num_model_builder.set_split()
    num_model_builder.train_model(RandomForestClassifier(10))
    predictions = num_model_builder.test_model()
    num_model_builder.eval_model(predictions)

    # stdev model
    target_column_name="SD_LOG(RATE_USD)"
    input_df = oa_collapsed_orders.drop(columns=[
        "ORDER_OFFER_AMOUNT",
        "LOG(RATE_USD)",
        "LOG(ORDER_OFFER_AMOUNT)"
    ])
    predicted_avg_column = avg_model_builder.model.predict(
        input_df[avg_model_builder.top_n_correlated_selected_column_names])
    input_df["PRED_LOG(RATE_USD)"] = predicted_avg_column

    predicted_num_column = num_model_builder.model.predict(
        input_df[num_model_builder.top_n_correlated_selected_column_names])
    input_df["PRED_ORDER_OFFER_AMOUNT"] = predicted_num_column

    sd_model_builder = util.Temp_Order_Classification_Model_Builder(
        df=input_df,
        target_column_name=target_column_name,
        weight_column_name=weight_column_name
    )
    sd_model_builder.transform_y(inplace=True)
    sd_model_builder.transform_X(inplace=True)
    sd_model_builder.set_split()
    sd_model_builder.train_model(RandomForestClassifier(10))
    predictions = sd_model_builder.test_model()
    sd_model_builder.eval_model(predictions)

    # building output file

    predicted_sd_column = sd_model_builder.model.predict(
        input_df[sd_model_builder.top_n_correlated_selected_column_names])
    input_df["PRED_SD_LOG(RATE_USD)"] = predicted_sd_column

    output_df = pd.DataFrame()
    output_df["REFERENCE_NUMBER"] = oa["REFERENCE_NUMBER"]
    output_df["PREDICTED_LOG_AVG"] = input_df["PRED_LOG(RATE_USD)"]
    output_df["PREDICTED_STDEV"] = input_df["PRED_SD_LOG(RATE_USD)"].apply(
        (lambda x: ([0]+sd_model_builder.percentiles)[int(x)])
    )
    # writing output file

    path_file_temp_avg_stdev = os.path.join(path_folder_data_temp, file_name_temp_avg_stdev)
    output_df.to_csv(path_file_temp_avg_stdev,index=False)
    return output_df
def main(args):
    path_folder = args["path_folder"]
    file_name = args["file_name_temp_avg_stdev"]
    output_df = dataclean(path_folder, file_name)

# added ftl removal and leadtime toggle
# - no ftl_only, no leadtime = avg=84%, num=60% stdev=67%
# - ftl_only, no leadtime = avg=91%, num=59%, stdev=65%
# - ftl_only, leadtime = avg=91%, num=60%, stdev=65%
# - no ftl_only, leadtime = avg=81%, num=60%, stdev=67%

# included ohe column name dropping in the for loop for ohe column creation (to automate the ftl filtering for transport_mode)
# ensuring that the oa_collapse_orders.drop(columns=) for categorical columns are no longer needed
# (note that this means that month and day columns are removed)
# removed the weight_column_name lines during the temp_order_model_builder cells

