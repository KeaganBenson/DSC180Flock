import sys
import src.Clean as Clean
import src.Model as Model
import src.SecretaryMethod as SecretaryMethod 
import os
    
def main_test(path_folder = "../../test/test-data/"):
    offers = pd.read_csv("offer_acceptance_offers.csv",low_memory=False)
    orders = pd.read_csv('offer_acceptance_orders.csv',low_memory=False)
    orders = Clean.clean_orders(orders)
    offers = Clean.clean_offers(offers)
    train,test = Clean.create_train_test_orders(orders)
    joinedDF = Clean.createJoinedDF(offers,train)
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
    x = numerical_loggable_column_names + bool_column_names 
    NUMBER_OFFERSModel = Model.Train_model(joinedDF,x,'NUMBER_OFFERS')
    testDF = Model.PrepareTestSet(test,NUMBER_OFFERSModel,x,'EstimatedNumOffs')
    results = SecretaryMethod.runSecretaryMethod(testDF['REFERENCE_NUMBER'],testDF,offers)
    final = SecretaryMethod.createFinalDF(offers,results)
    return final

def main(targets):
    main_path_folder = os.path.join(os.getcwd(),"data")
    test_path_folder = os.path.join(os.getcwd(),"test","test-data",'raw')
    path_folder = main_path_folder
    for target in targets:
        if target in ["test"]:
            path_folder = test_path_folder
            main_test(path_folder)
        
if __name__ == "__main__":
    targets = sys.argv[1:]
    #target = "test"
    main(targets)