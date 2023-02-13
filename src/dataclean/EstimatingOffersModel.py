#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
path_folder = "../../"
import sys
sys.path.insert(0, path_folder+"/src/")




import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def estimating_offers_model(path_file_offers, path_file_orders):
    pd.options.display.max_columns = 31

    #path_file_orders is the file path for the orders data
    #path_file_offers is the file path for the offers data

    offers = pd.read_csv(path_file_offers,low_memory=False)
    orders = pd.read_csv(path_file_orders,low_memory=False)

    # variables used for modeling
    bool_column_names = [
     'FD_ENABLED',
     'EXCLUSIVE_USE_REQUESTED',
     'HAZARDOUS',
     'REEFER_ALLOWED',
     'STRAIGHT_TRUCK_ALLOWED',
     'LOAD_TO_RIDE_REQUESTED',
    ]

    numerical_loggable_column_names = [
     'APPROXIMATE_DRIVING_ROUTE_MILEAGE',
     'PALLETIZED_LINEAR_FEET',
     'SECONDS_BETWEEN_ORDER_AND_DEADLINE',
     'LOAD_BAR_COUNT',
     'ESTIMATED_COST_AT_ORDER'
    ]

    # dates -> DT & creating new time delta variable
    orders['ORDER_DATETIME_PST'] = pd.to_datetime(orders['ORDER_DATETIME_PST'])
    orders['PICKUP_DEADLINE_PST'] = pd.to_datetime(orders['PICKUP_DEADLINE_PST'])
    offers['CREATED_ON_HQ'] = pd.to_datetime(offers['CREATED_ON_HQ'])
    orders['Time_between_Order_pickup'] = (orders['PICKUP_DEADLINE_PST'] - 
                                                    orders['ORDER_DATETIME_PST'])

    # fill driving distance with mean of in zip code results, most na were in zip code
    same = orders[orders['ORIGIN_3DIGIT_ZIP'] ==  orders['DESTINATION_3DIGIT_ZIP']]
    mean_same_distance = same['APPROXIMATE_DRIVING_ROUTE_MILEAGE'].dropna().mean()
    orders['APPROXIMATE_DRIVING_ROUTE_MILEAGE'] = orders['APPROXIMATE_DRIVING_ROUTE_MILEAGE'].fillna(mean_same_distance)

    # fill FD enabled with false
    orders['FD_ENABLED'] = orders['FD_ENABLED'].fillna(False)

    # drop everything else, cause all had consistance NA 7
    # 18 in Transportmode
    # change all boolean data back to boolean
    containsNa = (orders.isnull().sum() > 1).to_dict()
    orders = orders.dropna()
    for key,value in containsNa.items():
        if value:
            if orders[key].unique().sum() == 1:
                orders[key] = orders[key].astype(bool)

    s = offers.groupby('REFERENCE_NUMBER').count()['CARRIER_ID']
    nOffers_rec = s[s < 15]

    ftl = orders[orders['TRANSPORT_MODE'] == 'FTL']
    ftl.set_index('REFERENCE_NUMBER',inplace = True)
    joinedDF = ftl.join(nOffers_rec,how = 'left')
    joinedDF['NUMBER_OFFERS'] = joinedDF['CARRIER_ID'].fillna(0)
    joinedDF.drop('CARRIER_ID',axis = 1,inplace = True)
    joinedDF['SECONDS_BETWEEN_ORDER_AND_DEADLINE'] = joinedDF['Time_between_Order_pickup'].dt.total_seconds()

    accept_offers = offers[offers['LOAD_DELIVERED_FROM_OFFER'] == True]

    joined_oo = pd.merge(accept_offers, orders, how='inner')

    joined_oo['LEAD_TIME'] =  joined_oo['PICKUP_DEADLINE_PST'] - joined_oo['CREATED_ON_HQ']

    #joined table of both ftl/ptl offers that are accepeted, column LEAD_TIME is time between accpt offr and pickupdeadline
    joined_oo['LEAD_TIME'].describe()

    def bin_leadtime(df):
        if df['LEAD_TIME'] >= (df.quantile(q=0.75)['LEAD_TIME']).astype('dateime'):
            return 4
        elif df['LEAD_TIME'] >= (df.quantile(q=0.5)['LEAD_TIME']).astype('dateime'):
            return 3
        elif df['LEAD_TIME'] >= (df.quantile(q=0.25)['LEAD_TIME']).astype('dateime'):
            return 2
        return 1

    # somethings wrong with these 2 lines
    #joined_oo['LEADTIME_BIN'] = joined_oo.apply(bin_leadtime, axis=1)
    #joined_oo.quantile(q=0.75)
    print("joined_oo.apply(bin_leadtime, axis=1) runs into an error; running bin_leadtime will be skipped")

    train=joinedDF.sample(frac=0.8,random_state=200)
    test=joinedDF.drop(train.index)

    bool_column_names + numerical_loggable_column_names

    xTrain = train[bool_column_names + numerical_loggable_column_names].to_numpy()
    yTrain = train['NUMBER_OFFERS'].to_numpy()
    xTest = test[bool_column_names + numerical_loggable_column_names].to_numpy()
    yTest = test['NUMBER_OFFERS'].to_numpy()

    reg = LinearRegression().fit(xTrain, yTrain)
    preds = reg.predict(xTest).round()
    mean_absolute_error(yTest,preds)

    return reg, joinedDF[bool_column_names + numerical_loggable_column_names]

def main(args):
    path_folder = args["path_folder"]
    #file_name_temp_amount = args["file_name_temp_amount"]
    #file_name_output_df = "temp_amount.csv"

    path_folder_data = os.path.join(path_folder, "data")
    path_folder_data_raw = os.path.join(path_folder_data, "raw")
    path_file_orders = os.path.join(path_folder_data_raw, "offer_acceptance_orders.csv")
    path_file_offers = os.path.join(path_folder_data_raw, "offer_acceptance_offers.csv")
    
    model, joinedDF = estimating_offers_model(path_file_offers, path_file_orders)

    predictions = model.predict(joinedDF.to_numpy())
    # making a dataframe with the reference number and predicted offer amount
    output_df = pd.DataFrame()
    joinedDF.reset_index(drop=False, inplace=True)

    output_df["REFERENCE_NUMBER"] = joinedDF["REFERENCE_NUMBER"]
    # predicted amount (of offers)
    output_df["PREDICTED_AMOUNT"] = predictions

    # saving the output df to the data/temp folder as a csv under the name "temp_amount.csv"
    file_name_output_df = "temp_amount.csv"
    path_file_output_df = os.path.join(path_folder_data, "temp", file_name_output_df)
    output_df.to_csv(path_file_output_df, index=False)


# In[3]:


main({
    "path_folder": "../../"
})


# In[ ]:




