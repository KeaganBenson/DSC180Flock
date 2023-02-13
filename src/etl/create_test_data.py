#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd

def create_test_data(args):
    path_folder_data = args["path_folder_data"]
    w = 50
    h = 25
    
    path_folder_data_raw = os.path.join(path_folder_data, "raw")
    path_file_zipcode_coordinates =  os.path.join(path_folder_data_raw, "zipcode_coordinates.csv")
    path_file_oa_orders = os.path.join(path_folder_data_raw, "offer_acceptance_orders.csv")
    path_file_oa_offers = os.path.join(path_folder_data_raw, "offer_acceptance_offers.csv")
    
    test_zipcode_coordinates1 = pd.DataFrame()
    test_zipcode_coordinates1["3DIGIT_ZIP"] = pd.Series((np.arange(50)).astype(int))#.astype(str).str.zfill(3)
    test_zipcode_coordinates1["X_COORD"] = 0
    test_zipcode_coordinates1["Y_COORD"] = (np.arange(w))*1.0e7

    test_zipcode_coordinates2 = pd.DataFrame()
    test_zipcode_coordinates2["3DIGIT_ZIP"] = pd.Series((np.arange(50)+50).astype(int))#.astype(str).str.zfill(3)
    test_zipcode_coordinates2["X_COORD"] = 10
    test_zipcode_coordinates2["Y_COORD"] = (np.arange(w))*1.0e7

    test_zipcode_coordinates = pd.concat(
        [test_zipcode_coordinates1, test_zipcode_coordinates2],axis=0)

    test_oa_orders = pd.DataFrame()
    test_oa_orders["REFERENCE_NUMBER"] = ['[\n  "{0}"\n]'.format("a"+str(x)) for x in range(w)]
    test_oa_orders["ORIGIN_3DIGIT_ZIP"] =  pd.Series((np.arange(w)).astype(int))#.astype(str).str.zfill(3)
    test_oa_orders["DESTINATION_3DIGIT_ZIP"] = pd.Series((np.arange(w)+w).astype(int))#.astype(str).str.zfill(3)
    test_oa_orders["TRANSPORT_MODE"] = ["FTL" for _ in range(h)] + ["PTL" for _ in range(h)]
    test_oa_orders["STRAIGHT_TRUCK_ALLOWED"] = [True for _ in range(h)] + [False for _ in range(h)]
    test_oa_orders["LOAD_TO_RIDE_REQUESTED"] = [True for _ in range(h)] + [False for _ in range(h)]
    test_oa_orders["REEFER_ALLOWED"] = [True for _ in range(h)] + [False for _ in range(h)]
    test_oa_orders["HAZARDOUS"] = [True for _ in range(h)] + [False for _ in range(h)]
    test_oa_orders["EXCLUSIVE_USE_REQUESTED"] = [True for _ in range(h)] + [False for _ in range(h)]
    test_oa_orders["FD_ENABLED"] = [True for _ in range(h)] + [False for _ in range(h)]
    test_oa_orders["ESTIMATED_COST_AT_ORDER"] = np.log1p(np.arange(w))
    test_oa_orders["LOAD_BAR_COUNT"] = np.log1p(np.arange(w))

    test_oa_orders["LOAD_BAR_COUNT"] = np.log1p(np.arange(w))
    test_oa_orders["PALLETIZED_LINEAR_FEET"] = (np.arange(w))+1
    test_oa_orders["ORDER_DATETIME_PST"] = ['2022-09-07 10:07:05',
     '2022-05-19 09:21:39',
     '2022-07-06 09:45:30',
     '2022-09-12 05:38:14',
     '2022-06-21 10:52:48',
     '2022-08-30 13:25:32',
     '2022-06-17 08:33:54',
     '2022-05-27 08:06:34',
     '2022-06-02 14:40:50',
     '2022-10-08 13:41:18',
     '2022-08-22 14:00:31',
     '2022-10-13 10:51:58',
     '2022-07-26 14:23:50',
     '2022-10-04 09:06:28',
     '2022-08-24 09:23:26',
     '2022-05-16 11:57:28',
     '2022-07-21 05:16:48',
     '2022-09-16 11:14:27',
     '2022-05-05 06:51:53',
     '2022-05-18 11:19:03',
     '2022-09-22 14:45:40',
     '2022-08-19 14:00:11',
     '2022-09-22 07:30:34',
     '2022-09-27 09:51:31',
     '2022-09-16 13:13:01',
     '2022-08-26 06:57:03',
     '2022-08-12 06:04:22',
     '2022-06-14 08:34:31',
     '2022-07-28 13:51:57',
     '2022-06-14 09:40:38',
     '2022-07-12 14:40:53',
     '2021-12-15 11:25:20',
     '2021-12-02 04:26:25',
     '2021-09-23 09:09:28',
     '2022-06-14 12:14:54',
     '2022-05-26 09:40:38',
     '2022-02-17 10:47:24',
     '2020-09-18 10:46:49',
     '2020-02-17 11:52:56',
     '2021-09-02 12:11:33',
     '2020-08-26 08:51:50',
     '2022-03-22 20:58:55',
     '2019-11-01 06:32:43',
     '2022-04-19 12:47:31',
     '2019-11-11 08:12:11',
     '2019-11-11 09:33:23',
     '2020-02-04 13:34:43',
     '2020-01-03 10:44:29',
     '2022-05-26 10:44:58',
     '2019-09-25 08:49:02']
    test_oa_orders["PICKUP_DEADLINE_PST"] = ['2022-09-12 11:00:00',
     '2022-05-24 13:00:00',
     '2022-07-08 14:00:00',
     '2022-09-19 10:30:00',
     '2022-07-05 13:00:00',
     '2022-09-02 15:00:00',
     '2022-06-27 14:00:00',
     '2022-05-31 16:00:00',
     '2022-06-07 13:00:00',
     '2022-10-17 13:00:00',
     '2022-08-23 12:00:00',
     '2022-10-19 11:00:00',
     '2022-10-11 12:00:00',
     '2022-10-05 17:00:00',
     '2022-08-29 12:00:00',
     '2022-05-19 11:00:00',
     '2022-07-21 15:00:00',
     '2022-09-19 16:00:00',
     '2022-05-13 09:00:00',
     '2022-05-20 14:00:00',
     '2022-10-03 14:00:00',
     '2022-08-25 16:00:00',
     '2022-09-27 10:00:00',
     '2022-09-29 18:00:00',
     '2022-09-19 13:00:00',
     '2022-08-30 09:30:00',
     '2022-08-15 11:30:00',
     '2022-06-17 14:00:00',
     '2022-07-29 12:00:00',
     '2022-06-20 10:00:00',
     '2022-07-14 13:00:00',
     '2021-12-16 15:00:00',
     '2021-12-02 16:30:00',
     '2021-09-27 16:00:00',
     '2022-06-16 14:00:00',
     '2022-05-27 17:00:00',
     '2022-02-17 17:30:00',
     '2020-09-18 16:00:00',
     '2020-02-17 16:00:00',
     '2021-09-02 16:30:00',
     '2020-08-26 15:00:00',
     '2022-03-24 14:00:00',
     '2019-11-11 09:00:00',
     '2022-04-22 13:00:00',
     '2019-11-11 13:30:00',
     '2019-11-12 13:00:00',
     '2020-02-11 16:30:00',
     '2020-01-07 14:30:00',
     '2022-05-26 16:00:00',
     '2019-09-25 16:00:00']
    test_oa_orders["DELIVERY_TIME_CONSTRAINT"] = ["SCHEDULE" for _ in range(h)] + ["NONE" for _ in range(h)]
    test_oa_orders = pd.concat([test_oa_orders.iloc[:h],test_oa_orders.iloc[-h:]], axis=0)
    test_oa_orders["APPROXIMATE_DRIVING_ROUTE_MILEAGE"] = (
        np.log1p(np.arange(w))
    )
    
    test_oa_offers = pd.DataFrame()
    test_oa_offers["RATE_USD"] = np.log1p(np.arange(w))
    boolean_column_names = ['SELF_SERVE', 'IS_OFFER_APPROVED',
           'AUTOMATICALLY_APPROVED', 'MANUALLY_APPROVED', 'WAS_EVER_UNCOVERED',
           'COVERING_OFFER', 'LOAD_DELIVERED_FROM_OFFER', 'RECOMMENDED_LOAD',
           'VALID']
    for column_name in boolean_column_names:
        test_oa_offers[column_name] = [True for _ in range(h)] + [False for _ in range(h)]
    test_oa_offers["OFFER_TYPE"] = ["pool" for _ in range(h)] + ["quote" for _ in range(h)]
    test_oa_offers["REFERENCE_NUMBER"] = ['[\n  "{0}"\n]'.format("a"+str(x)) for x in range(w)]
    test_oa_offers["CARRIER_ID"] = [("a"+str(x)) for x in range(w)]
    test_oa_offers["CREATED_ON_HQ"] = ['2021-11-03 08:54:13',
     '2021-11-03 08:57:27',
     '2021-11-03 08:57:28',
     '2021-11-03 08:57:40',
     '2021-11-03 09:05:43',
     '2022-07-26 15:50:19',
     '2022-07-26 15:59:44',
     '2022-07-27 09:33:05',
     '2022-08-19 09:39:41',
     '2022-09-09 07:58:31',
     '2022-08-30 09:15:39',
     '2022-08-30 09:21:56',
     '2022-08-30 09:26:47',
     '2021-07-12 07:53:40',
     '2021-09-13 04:46:56',
     '2021-08-31 07:26:42',
     '2021-09-13 09:21:22',
     '2021-09-16 11:47:43',
     '2021-10-18 15:29:39',
     '2021-10-18 15:33:59',
     '2022-06-30 05:11:13',
     '2022-06-29 07:33:36',
     '2022-06-29 11:47:41',
     '2022-06-30 05:00:56',
     '2022-06-30 05:12:08',
     '2022-06-29 10:05:24',
     '2022-06-29 07:42:23',
     '2022-06-29 07:43:25',
     '2022-06-29 11:54:20',
     '2022-03-09 11:22:39',
     '2021-08-25 07:35:41',
     '2022-03-04 12:14:04',
     '2022-03-04 12:22:10',
     '2022-03-04 12:22:10',
     '2021-12-01 08:25:17',
     '2021-12-01 09:55:39',
     '2021-12-01 11:53:58',
     '2021-12-01 08:25:34',
     '2021-12-01 11:39:50',
     '2022-03-31 08:35:20',
     '2022-03-31 08:52:40',
     '2022-03-31 08:24:48',
     '2022-03-31 08:28:17',
     '2022-03-31 08:58:04',
     '2022-03-31 09:06:00',
     '2022-03-31 09:11:34',
     '2022-03-31 08:28:23',
     '2022-03-31 09:11:54',
     '2022-09-27 14:51:03',
     '2022-09-27 13:57:22']
    test_zipcode_coordinates.to_csv(path_file_zipcode_coordinates,index=False)
    test_oa_orders.to_csv(path_file_oa_orders,index=False)
    test_oa_offers.to_csv(path_file_oa_offers,index=False)

def main(args):
    create_test_data(args)

