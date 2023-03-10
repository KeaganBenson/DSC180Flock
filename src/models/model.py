#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
def model(args):
    print('Starting the Model')
    path_folder_data = args["path_folder_data"]
    file_name_final_df = args["file_name_final_df"]

    path_folder_data_raw = os.path.join(path_folder_data, "raw")
    path_folder_data_out = os.path.join(path_folder_data, "out")

    path_file_oa_orders = os.path.join(path_folder_data_raw, "offer_acceptance_orders.csv")
    path_file_oa_offers = os.path.join(path_folder_data_raw, "offer_acceptance_offers.csv")
    oa_orders = pd.read_csv(path_file_oa_orders)
    oa_offers = pd.read_csv(path_file_oa_offers)

    path_file_final_df = os.path.join(path_folder_data_out, file_name_final_df)
    output = pd.read_csv(path_file_final_df)

    oa_orders = oa_orders[oa_orders["TRANSPORT_MODE"]=="FTL"]

    full = oa_orders.merge(oa_offers,on=["REFERENCE_NUMBER"])

    final = output.set_index('REFERENCE_NUMBER').join(full.set_index('REFERENCE_NUMBER'),how='inner').dropna()
    final.reset_index(inplace = True)
    idx = final.groupby('REFERENCE_NUMBER')['RATE_USD'].idxmin()
    final['GoodOffer'] = final.index.isin(idx) | final['LOAD_DELIVERED_FROM_OFFER']
    num_features = ['PREDICTED_AMOUNT','PREDICTED_LOG_AVG','PREDICTED_STDEV',
                    'APPROXIMATE_DRIVING_ROUTE_MILEAGE', 'RATE_USD','RECOMMENDED_LOAD']
    final.set_index('REFERENCE_NUMBER',inplace = True)
    final['index'] = final.groupby('REFERENCE_NUMBER')['CREATED_ON_HQ'].rank(method = 'dense')
    final = final.reset_index().set_index(['REFERENCE_NUMBER','index'])
    test = final[final['ORDER_DATETIME_PST'] > '2022-07-01']

    indexes_to_keep = final.index.difference(test.index)
    train = final.loc[indexes_to_keep]
    X_train = train[num_features]
    X_test = test[num_features]
    y_train = train['GoodOffer']
    y_test = test['GoodOffer']


    defaultDance = X_test.copy()
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # predict the target values for the testing data
    y_pred = clf.predict_proba(X_test)[:,1]
    defaultDance['Pred'] = (y_pred >=.40)
    defaultDance['Real'] = y_test

    lfg = defaultDance.join(final['LOAD_DELIVERED_FROM_OFFER'],how='left')[['RATE_USD','Pred','LOAD_DELIVERED_FROM_OFFER']]
    df_first = lfg[lfg['Pred']].groupby(level='REFERENCE_NUMBER')['Pred'].idxmin().apply(lambda x: lfg.loc[x])

    us = lfg[lfg['Pred']].groupby('REFERENCE_NUMBER').head(1)['RATE_USD']
    flock =lfg[lfg['LOAD_DELIVERED_FROM_OFFER']].groupby('REFERENCE_NUMBER').head(1)['RATE_USD']
    #OUR MODEL STAT
    print('Our Model ---')
    print('Average Rate ' + str(us.mean()))
    print('Number of Orders Taken ' + str(len(us)))
    #Flock STATs
    print("Flock's Result ---")
    print('Average Rate ' + str(flock.mean()))
    print('Number of Orders Taken ' + str(len(flock)))
    
    print('Difference ---')
    print('Average Rate was ' + str( round(abs(us.mean()-flock.mean())/flock.mean() *100,2) ) + '% better')

def main(args):
    model(args)


# In[ ]:




