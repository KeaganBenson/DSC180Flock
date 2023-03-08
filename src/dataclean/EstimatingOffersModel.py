#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
#path_folder = "../../"
#import sys
#sys.path.insert(0, path_folder+"/src/")




import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def estimating_offers_clean(path_file_offers, path_file_orders):
    offers = pd.read_csv(path_file_offers,low_memory=False)
    orders = pd.read_csv(path_file_orders,low_memory=False)
    # dates -> DT & creating new time delta variable
    orders['ORDER_DATETIME_PST'] = pd.to_datetime(orders['ORDER_DATETIME_PST'])
    orders['PICKUP_DEADLINE_PST'] = pd.to_datetime(orders['PICKUP_DEADLINE_PST'])
    offers['CREATED_ON_HQ'] = pd.to_datetime(offers['CREATED_ON_HQ'])
    orders['Time_between_Order_pickup'] = (orders['PICKUP_DEADLINE_PST'] - 
                                                    orders['ORDER_DATETIME_PST'])
    orders['SECONDS_BETWEEN_ORDER_AND_DEADLINE'] = orders['Time_between_Order_pickup'].dt.total_seconds()

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
    return [orders,offers]

def calc_business_days(row):
    lead = pd.bdate_range(start=row['OrderDate'], end=row['PickupDate'])
    offer = pd.bdate_range(start=row['OrderDate'], end=row['OfferDate'])
    return len(lead),len(offer)

def createWeightVector(orders,offers):
    acc = offers.set_index('REFERENCE_NUMBER')
    acc = acc[acc['LOAD_DELIVERED_FROM_OFFER']]
    full2 = orders.set_index('REFERENCE_NUMBER').join(acc,how = 'inner')
    dates = full2[['ORDER_DATETIME_PST','PICKUP_DEADLINE_PST','CREATED_ON_HQ']].copy()
    dates['OrderDate'] =dates['ORDER_DATETIME_PST'].dt.date
    dates['PickupDate'] = dates['PICKUP_DEADLINE_PST'].dt.date
    dates['OfferDate'] = dates['CREATED_ON_HQ'].dt.date
    s = dates.apply(calc_business_days, axis=1)
    dates[['Lead_B_days', 'Offer_B_days']] = s.apply(lambda x: pd.Series([x[0], x[1]]))
    dates['orderHours'] = 21 - dates['ORDER_DATETIME_PST'].dt.hour
    dates['pickupHours'] = dates['PICKUP_DEADLINE_PST'].dt.hour - 7
    dates['offerHours'] = dates['CREATED_ON_HQ'].dt.hour - 5
    dates['leadHours'] = (dates['Lead_B_days'] - 2)*16 + dates['orderHours'] + dates['pickupHours']
    dates['OfferOrderHours'] = (dates['Offer_B_days'] - 2)*16 + dates['orderHours'] + dates['offerHours']
    dates['Weight'] = dates['OfferOrderHours']/dates['leadHours']
    return dates['Weight']

def estimating_offers_model(path_file_offers, path_file_orders):
    df = estimating_offers_clean(path_file_offers, path_file_orders)
    orders = df[0]
    offers = df[1]
    ftl = orders[orders['TRANSPORT_MODE']=='FTL']

    weightVector = createWeightVector(ftl,offers)

    full = ftl.set_index('REFERENCE_NUMBER').join(offers.set_index('REFERENCE_NUMBER'),how = 'inner')
    full['NumberOffers'] = 1
    num_offer = full.groupby('REFERENCE_NUMBER').count()['NumberOffers'].clip(upper=20)
    new = ftl.set_index('REFERENCE_NUMBER').join(weightVector,how='inner').join(num_offer,how='inner')
    new = new[new['Weight'] > 0]
    new = new[new['Weight'] < 1].copy()

    bool_column_names = [
        'FD_ENABLED',
        'EXCLUSIVE_USE_REQUESTED',
        'HAZARDOUS',
        'REEFER_ALLOWED',
        'STRAIGHT_TRUCK_ALLOWED',
        'LOAD_TO_RIDE_REQUESTED'
    ]
    numerical_column_names = [
        'APPROXIMATE_DRIVING_ROUTE_MILEAGE',
        'PALLETIZED_LINEAR_FEET',
        'SECONDS_BETWEEN_ORDER_AND_DEADLINE',
        'LOAD_BAR_COUNT',
        'ESTIMATED_COST_AT_ORDER',
    ]
    cat_column = [
        'DELIVERY_TIME_CONSTRAINT',
        'ORIGIN_3DIGIT_ZIP',
        'DESTINATION_3DIGIT_ZIP',
    ]
    X_train, X_test, y_train, y_test = train_test_split(new.loc[:,new.columns!='NumberOffers'], new['NumberOffers'], test_size=0.2, random_state=42)
    # Create a preprocessor to scale numerical features and one hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_column),
            ('num', StandardScaler(), numerical_column_names),
            ('bool','passthrough',bool_column_names),
            ])

    # Fit the column transformer to the training data
    preprocessor.fit(X_train)

    # Transform the training and testing data using the column transformer
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Train a logistic regression model using the transformed data
    model = LinearRegression()
    model.fit(X_train_transformed, y_train,sample_weight = X_train['Weight'])

    y_pred = model.predict(X_test_transformed).round()
    mean_absolute_error(y_test,y_pred)

    return model, new[bool_column_names + numerical_column_names + cat_column]

def main(args):
    path_folder_data = args["path_folder_data"]
    #file_name_temp_amount = args["file_name_temp_amount"]
    #file_name_output_df = "temp_amount.csv"

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
    
    return output_df


# In[3]:

# In[ ]:




