import pandas as pd

def create_train_test_orders(orders):
    train = orders.sample(frac=0.98,random_state=200)
    test = orders.drop(train.index)
    return train, test

def clean_orders(orders):
    # converting time to dt and creating time difference
    
    orders['ORDER_DATETIME_PST'] = pd.to_datetime(orders['ORDER_DATETIME_PST'])
    orders['PICKUP_DEADLINE_PST'] = pd.to_datetime(orders['PICKUP_DEADLINE_PST'])
    orders['Time_between_Order_pickup'] = (orders['PICKUP_DEADLINE_PST'] - orders['ORDER_DATETIME_PST'])
    
    # filling nulls with mean square distance
    same = orders[orders['ORIGIN_3DIGIT_ZIP']==orders['DESTINATION_3DIGIT_ZIP']]
    mean_same_distance = same['APPROXIMATE_DRIVING_ROUTE_MILEAGE'].dropna().mean()
    orders['APPROXIMATE_DRIVING_ROUTE_MILEAGE'] = orders['APPROXIMATE_DRIVING_ROUTE_MILEAGE'].fillna(mean_same_distance)
    orders['SECONDS_BETWEEN_ORDER_AND_DEADLINE'] = orders['Time_between_Order_pickup'].dt.total_seconds()

    # fill FD enabled with false
    orders['FD_ENABLED'] = orders['FD_ENABLED'].fillna(False)
    orders = orders.dropna()
    
    # only FTL orders
    ftl = orders[orders['TRANSPORT_MODE'] == 'FTL']
    return ftl
    
def clean_offers(offers):
    offers['CREATED_ON_HQ'] = pd.to_datetime(offers['CREATED_ON_HQ'])
    offers = offers[offers['OFFER_TYPE'] == 'quote']
    return offers

def createJoinedDF(offers,ftl):
    numOffers_rec = offers.groupby('REFERENCE_NUMBER').count()['CARRIER_ID']
    averageRate = offers.groupby('REFERENCE_NUMBER').mean()['RATE_USD']
    numberOffers = numOffers_rec[numOffers_rec < 15]
    
    joinedDF = ftl.set_index('REFERENCE_NUMBER').join(numberOffers,how = 'inner')
    joinedDF = joinedDF.join(averageRate,how = 'inner')
    joinedDF['Average_Rate'] = joinedDF['RATE_USD'].fillna(0)
    joinedDF['NUMBER_OFFERS'] = joinedDF['CARRIER_ID'].fillna(0)
    joinedDF.drop('CARRIER_ID',axis = 1,inplace = True)
    joinedDF.drop('RATE_USD',axis = 1,inplace = True)
    return joinedDF
    