import numpy as np
import pandas as pd
def runSecretaryMethod(refNums,testSetdf,offers):
    offers = offers[['REFERENCE_NUMBER','RATE_USD','CREATED_ON_HQ','CARRIER_ID']]
    testDF = testSetdf[['REFERENCE_NUMBER','EstimatedNumOffs','ESTIMATED_COST_AT_ORDER']]
    
    ReferenceNumber = []
    rates = []
    carrier = []

    for i in range(len(refNums)):
        order = testSetdf[testSetdf['REFERENCE_NUMBER'] == refNums[i]]
        offs = offers[offers['REFERENCE_NUMBER'] == refNums[i]].sort_values(by='CREATED_ON_HQ', ascending=True)
        estimatedNumsOffer = order['EstimatedNumOffs'][i]
        secNum = round(estimatedNumsOffer/np.e)
        acceptedInSec = 0
        if len(offs) > 0:
            if estimatedNumsOffer == 1:
                record = offs.iloc[0]['RATE_USD']
                ReferenceNumber.append(refNums[i])
                rates.append(record)
                carrier.append(offs.iloc[0]['CARRIER_ID'])
            else:
                estimatedCost = order['ESTIMATED_COST_AT_ORDER'][i]
                for n in range(secNum-1):
                    actual_cost = offs.iloc[n]['RATE_USD']
                    if actual_cost < estimatedCost:
                        rates.append(actual_cost)
                        ReferenceNumber.append(refNums[i])
                        carrier.append(offs.iloc[0]['CARRIER_ID'])
                        acceptedInSec = 1
                        break
                if acceptedInSec == 0:
                    record = min(offs.iloc[:secNum]['RATE_USD'])
                    offerRate = offs.iloc[secNum:]['RATE_USD']
                    for num in offerRate:
                        if num < record:
                            ReferenceNumber.append(refNums[i])
                            rates.append(num)
                            carrier.append(offs.iloc[0]['CARRIER_ID'])
                            break
    return pd.DataFrame({'REFERENCE_NUMBER':ReferenceNumber,'Secretary_Carrier':carrier,'Secreatry_Rate':rates})

def createFinalDF(offers,results):
    offers = offers.set_index(['REFERENCE_NUMBER','CARRIER_ID'])
    flockAccept = pd.DataFrame(offers[offers['LOAD_DELIVERED_FROM_OFFER']].RATE_USD).reset_index()
    flockAccept = flockAccept.rename(columns = {'CARRIER_ID':'Flock_Carrier','RATE_USD':'Flock_Rate'})
    merged = results.merge(flockAccept,on='REFERENCE_NUMBER')
    merged['Difference'] = merged['Secreatry_Rate'] - merged['Flock_Rate']
    return merged
    
