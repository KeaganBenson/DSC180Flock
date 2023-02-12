#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import gdown
import pandas as pd
#import requests



def download_csv_from_google_drive_share_link_to_file_path(url, filepath):
    gdown.download(url=url, output=filepath, quiet=False, fuzzy=True)

def get_offer_acceptance_file_path(path_folder, filename):
    path_file = os.path.join(path_folder, filename)
    return path_file

def test_data_creator(path_folder):
    filename_offer_acceptance_orders = "offer_acceptance_orders.csv"
    
    filename_offer_acceptance_offers = "offer_acceptance_offers.csv"
    
    path_file_offer_acceptance_orders = get_offer_acceptance_file_path(path_folder, filename_offer_acceptance_orders)
    path_file_offer_acceptance_offers = get_offer_acceptance_file_path(path_folder, filename_offer_acceptance_offers)
    
    df = pd.read_csv(path_file_offer_acceptance_orders)
    df = df.head(100)
    df["TRANSPORT_MODE"] = "PTL"
    df["ORIGIN_3DIGIT_ZIP"] = df["ORIGIN_3DIGIT_ZIP"].astype(str).str.zfill(3)
    df["DESTINATION_3DIGIT_ZIP"] = df["DESTINATION_3DIGIT_ZIP"].astype(str).str.zfill(3)

    df["ORIGIN_3DIGIT_ZIP"][0] = "00a"
    df["DESTINATION_3DIGIT_ZIP"][0] = "00a"

    df["REFERENCE_NUMBER"] = ['''[a{0}]'''.format(str(i//2)) for i in range(100)]
    df.to_csv(path_file_offer_acceptance_orders, index=False)

    df = pd.read_csv(path_file_offer_acceptance_offers)
    df = df.head(100)
    df["LOAD_DELIVERED_FROM_OFFER"] = True
    df["REFERENCE_NUMBER"] = ['''[a{0}]'''.format(str(i//2)) for i in range(100)]
    df.to_csv(path_file_offer_acceptance_offers, index=False)

    
def main(path_folder = "../../data/raw"):
    
    offer_acceptance_offers_google_drive_url = "https://drive.google.com/file/d/17FWCWqGkhIwM7nHHLUNo1Ej2GnEJebmO/view?usp=sharing"
    
    offer_acceptance_orders_google_drive_url = "https://drive.google.com/file/d/1s8RCVGPUXW4G3ZeK0FXPi-P5p6AKYi3Y/view?usp=sharing"
    
    filename_offer_acceptance_orders = "offer_acceptance_orders.csv"
    
    filename_offer_acceptance_offers = "offer_acceptance_offers.csv"
    
    path_file_offer_acceptance_orders = get_offer_acceptance_file_path(path_folder, filename_offer_acceptance_orders)
    path_file_offer_acceptance_offers = get_offer_acceptance_file_path(path_folder, filename_offer_acceptance_offers)
    download_csv_from_google_drive_share_link_to_file_path(offer_acceptance_offers_google_drive_url, path_file_offer_acceptance_offers)
    download_csv_from_google_drive_share_link_to_file_path(offer_acceptance_orders_google_drive_url, path_file_offer_acceptance_orders)
# In[ ]:




