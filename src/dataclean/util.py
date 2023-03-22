

# In[ ]:





# ## Utility Functions

# In[4]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib    
from matplotlib import cm

import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

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
from sklearn.model_selection import StratifiedKFold 
# fig ax bounding points
Y_UPPER_BOUND = 6.5
Y_LOWER_BOUND = 2.5
X_UPPER_BOUND = -7
X_LOWER_BOUND = -15

def bound_ax(ax):
    ax.set_ylim((Y_LOWER_BOUND, Y_UPPER_BOUND))
    ax.set_xlim((X_LOWER_BOUND,X_UPPER_BOUND))
    return ax

class Metro_Cluster:
    '''
    A class for making the columns that would one-hot-encode the zipcodes into metropolitan regions
    '''
    def __init__(
        self,
        df,
        x_column_name,
        y_column_name,
        value_column_name,
        cluster_method="kmeans",
        group_column_name = "GROUP",
        group_amount = 20,
    ):
        """
        
        """
        assert cluster_method in ["kmeans","dbscan"]
        self.value_column_name = value_column_name
        
        if cluster_method == "kmeans":
            np.random.seed(5)
            clusterer = KMeans(group_amount).fit(df)
            df[group_column_name] = np.array(clusterer.labels_)
        if cluster_method == "dbscan":
            eps=0.15
            self.eps = eps
            clusterer = DBSCAN(eps=eps).fit(df)
            df[group_column_name] = np.array(clusterer.labels_)+1
            print(len(set(clusterer.labels_)))
            print((set(clusterer.labels_)))
            group_amount = len(set(clusterer.labels_))
            df = df[df[group_column_name]!=0]
        self.df = df
        self.cluster_method = cluster_method
        self.x_column_name = x_column_name
        self.y_column_name = y_column_name
        self.clusterer = clusterer
        self.group_amount = group_amount
        self.group_column_name = group_column_name
        
    def plot_map(self, path_folder,custom_cmap="Spectral"):
        """
        
        """
        import matplotlib.patches as mpatches
        if self.group_amount == 20:
            custom_cmap = 'tab20'
            
        cmap = plt.cm.get_cmap(custom_cmap)
        colors = cmap(np.linspace(0, 1, self.group_amount))
        fig, ax = plt.subplots(figsize=(10,6))
        ax = bound_ax(ax)
        patches = []
        for i in range(len(colors)):
            temp_color = colors[i]
            temp_group = i
            temp_df = self.df[self.df[self.group_column_name]==temp_group]
            ax.scatter(
                temp_df[self.x_column_name],
                temp_df[self.y_column_name],
                c=np.array([temp_color]),
                label=str(temp_group),
                s=5,
                alpha=0.6,
            )
            patches.append(mpatches.Patch(color=temp_color, label=str(temp_group)))
        fig.suptitle("Zipcodes clustered into Groups i.e. Metropolitan Regions")
        ax.scatter(self.cluster_df[self.x_column_name], 
                    self.cluster_df[self.y_column_name],
                    edgecolors='black')
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()
        fig.savefig(os.path.join(path_folder,'generated_visualizations','metro_clusters.png'),bbox_inches='tight')
        return ax
            
         
    def build_cluster_df(
        self,
        centroid_as_metro = False
    ):
        '''
        
        '''
        cluster_df = (
            self.df
            .groupby([self.group_column_name],as_index=False)
            .agg({self.x_column_name: np.mean, 
                  self.y_column_name: np.mean})
        )
        self.cluster_df = cluster_df
        centroid_x_column_name = "CENTROID_X_COORD"
        centroid_y_column_name = "CENTROID_Y_COORD"

        centroid_x_column = cluster_df[self.x_column_name]
        centroid_y_column = cluster_df[self.y_column_name]
        cluster_df[centroid_x_column_name] = centroid_x_column
        cluster_df[centroid_y_column_name] = centroid_y_column

        if centroid_as_metro == True:
            # if centroid as metro is true,
            # then re-write the centroid not as the centerpoint of the cluster, but the most active zipcode (aka city)
            
            cluster_df.sort_values(self.group_column_name,inplace=True)
            temp_df = self.df.sort_values(self.value_column_name, ascending=False)
            temp_df = temp_df.drop_duplicates(subset=[self.group_column_name])
            temp_df.sort_values(self.group_column_name,inplace=True)
            temp_df_x_column = temp_df[self.x_column_name]
            temp_df_y_column = temp_df[self.y_column_name]

            assert (list(temp_df[self.group_column_name]) == list(cluster_df[self.group_column_name]))
            
            cluster_df[self.x_column_name] = temp_df_x_column.values
            cluster_df[self.y_column_name] = temp_df_y_column.values
        
        #self.plot_map()

        self.cluster_df = cluster_df
        self.centroid_y_column_name = centroid_y_column_name
        self.centroid_x_column_name = centroid_x_column_name

        return cluster_df

    def build_opt_pivot(
        self,
        temp_opt_pivot_column_name = "temp"
    ):
        cluster_df_melted = self.cluster_df.melt(
            id_vars=[self.group_column_name],
            value_vars=[self.x_column_name, self.y_column_name])

        cluster_df_melted.sort_values(["variable",self.group_column_name],inplace=True)

        cluster_df_melted[temp_opt_pivot_column_name] = (
            cluster_df_melted["variable"]+"_"+cluster_df_melted[self.group_column_name].astype(str)
        )
        opt_pivot = cluster_df_melted.set_index(cluster_df_melted[temp_opt_pivot_column_name])[["value"]].T
        opt_pivot_x_column_names = [column_name 
                                    for column_name in opt_pivot.columns 
                                    if self.x_column_name in column_name]
        opt_pivot_y_column_names = [column_name 
                                    for column_name in opt_pivot.columns 
                                    if self.y_column_name in column_name]
        self.opt_pivot = opt_pivot
        self.opt_pivot_x_column_names = opt_pivot_x_column_names
        self.opt_pivot_y_column_names = opt_pivot_y_column_names
        return opt_pivot

    def get_proximities(
        self,
        df, 
        x_column_name, 
        y_column_name,
        proximity_column_names,
        zero_division_preventer = 1,
    ):
        
        if self.cluster_method == "kmeans":
            zero_division_preventer = 1
            x_distances = (np.expand_dims(df[x_column_name].values,1) - self.opt_pivot[self.opt_pivot_x_column_names].values)**2
            y_distances = (np.expand_dims(df[y_column_name].values,1) - self.opt_pivot[self.opt_pivot_y_column_names].values)**2
            distances = x_distances + y_distances            
            proximities = 1/(distances+zero_division_preventer)
            proximities_sum = np.sum(proximities,axis=1,keepdims=True)
            proximities /= proximities_sum
            return pd.DataFrame(proximities,columns=proximity_column_names)
        elif self.cluster_method == "dbscan":
            zero_division_preventer = 1
            x_distances = (np.expand_dims(df[x_column_name].values,1) - self.opt_pivot[self.opt_pivot_x_column_names].values)**2
            y_distances = (np.expand_dims(df[y_column_name].values,1) - self.opt_pivot[self.opt_pivot_y_column_names].values)**2
            distances = x_distances + y_distances
            proximities = 1/(distances+zero_division_preventer)
            proximities_sum = np.sum(proximities,axis=1,keepdims=True)
            proximities /= proximities_sum
            return pd.DataFrame(proximities,columns=proximity_column_names[1:])  

def dataclean_ftl_duplicate_references(oa_orders):
    """
    Part of a greater data cleaning on removing incorrect FTL labels,
    Removes FTL rows that have reference numbers that show up more than once, as those rows are incorrectly labeled FTL
    The dataframe is taken apart into FTL vs Non-FTL; then the FTL rows are removed of duplicates, and both FTL and Non-FTL are joined back together

    Args:
    oa_orders (pd.DataFrame): the offers data
    
    Returns:
    pd.DataFrame: given dataframe with said rows removed
    
    Raises:
    AssertionError: if necessary column names aren't in the given dataframe
    """
    assert "TRANSPORT_MODE" in oa_orders.columns
    assert "REFERENCE_NUMBER" in oa_orders.columns
    oa_orders_ftl = oa_orders[oa_orders["TRANSPORT_MODE"]=="FTL"]
    oa_orders_non_ftl = oa_orders[oa_orders["TRANSPORT_MODE"]!="FTL"]
    oa_orders_ftl_unique = oa_orders_ftl.drop_duplicates(subset=["REFERENCE_NUMBER"],keep=False)
    oa_orders = pd.concat([oa_orders_non_ftl, oa_orders_ftl_unique],axis=0)
    oa_orders = oa_orders.reset_index(drop=True)
    assert (oa_orders[oa_orders["TRANSPORT_MODE"]=="FTL"]["REFERENCE_NUMBER"].value_counts()==1).all()
    return oa_orders
def dataclean_ftl_nonquote(oa):
    """
    Part of a greater data cleaning on removing incorrect FTL labels,
    Removes FTL rows that don't have quote as the transport mode

    Args:
    oa (pd.DataFrame): the join of the offers data and orders data
    
    Returns:
    pd.DataFrame: given dataframe with said rows removed
    
    Raises:
    AssertionError: if necessary column names aren't in the given dataframe
    """
    assert "OFFER_TYPE" in oa.columns
    assert "TRANSPORT_MODE" in oa.columns    
    oa = oa[~((oa["TRANSPORT_MODE"]=="FTL") & (oa["OFFER_TYPE"]!="quote"))]
    return oa
def get_list_of_reference_numbers_column(column):
    """
    Cleans the raw reference column from a string to a list of strings
    """
    result = (column
    ).str.replace("\n"," "
    ).str.replace(" ",""
    ).str.replace('''"''',''
    ).str.replace("[",""
    ).str.replace("]",""
    ).str.split(",")
    return result
def add_cleaned_reference_numbers_column(df,column_name="REFERENCE_NUMBERS"):
    """
    Adds column with reference numbers cleaned into a list.

    Args:
    df (pd.DataFrame): the given dataframe
    
    Returns:
    pd.DataFrame: given dataframe with said column added
    
    Raises:
    AssertionError: if necessary column names aren't in the given dataframe
    """
    assert "REFERENCE_NUMBER" in list(df.columns)
    cleaned_reference_number_column = get_list_of_reference_numbers_column(df["REFERENCE_NUMBER"])
    df[column_name] = cleaned_reference_number_column
    return df
def add_num_reference_numbers_column(oa_offers,column_name="NUM_REFERENCE_NUMBERS"):
    """
    Adds number of reference numbers by a carrier (used to find pooled)

    Args:
    df (pd.DataFrame): the given dataframe
    
    Returns:
    pd.DataFrame: given dataframe with said column added
    
    Raises:
    AssertionError: if necessary column names aren't in the given dataframe
    """
    assert "REFERENCE_NUMBERS" in list(oa_offers.columns)
    oa_offers[column_name] = oa_offers["REFERENCE_NUMBERS"].apply(len)
    return oa_offers
def dataclean_pool_reference_numbers_discrepancy(oa_offers):
    """
    Some oa_offers rows have a mismatch between the length of its reference numbers and its pooled/quote labelling
    These rows are wrong, and this function removes them

    Args:
    oa_offers (pd.DataFrame): the offers data
    
    Returns:
    pd.DataFrame: given dataframe with said rows removed
    
    Raises:
    AssertionError: if necessary column names aren't in the given dataframe
    """
    assert "OFFER_TYPE" in oa_offers.columns
    assert "NUM_REFERENCE_NUMBERS" in oa_offers.columns    
    oa_offers = oa_offers[((oa_offers["OFFER_TYPE"]=="pool") == (oa_offers["NUM_REFERENCE_NUMBERS"]>1))]
    return oa_offers  
def dataclean_filter_ftl(oa_orders):
    """
    Removes rows that aren't FTL
    Args:
    oa (pd.DataFrame): the orders data
    
    Returns:
    pd.DataFrame: given dataframe with said rows removed
    
    Raises:
    AssertionError: if necessary column names aren't in the given dataframe
    """
    assert "TRANSPORT_MODE" in list(oa_orders.columns)
    oa_orders = oa_orders[oa_orders["TRANSPORT_MODE"]=="FTL"]
    print("removing rows in dataclean_filter_ftl")
    return oa_orders





def add_columns(df,columns):
    """
    Horizontally Concatenates a smaller dataframe (columns) to a bigger one (df)
    
    Args:
    oa (pd.DataFrame): the orders data
    
    Returns:
    pd.DataFrame: given dataframe with said rows removed
    
    Raises:
    AssertionError: if necessary column names aren't in the given dataframe
    """
    df = pd.concat([df,columns],axis=1)
    return df

## One hot encoding
##def temp_build_ohe(categorical_column, column_name):
##    '''
##    Builds the categorical_column
##    Args:
##    categorical_column (pd.Series): the categorical column that is to be one-hot-encoded
##    column_name (str): the column name of the categorical column that is to be one-hot-encoded, and to be the prefix of the new one-hot-encoded columns
##    drop_preohe (bool): boolean to decide to drop the categorical column from df
##    
##    Returns:
##    pd.DataFrame, the one-hot-encoded columns, of column names with the format "column_name=class_1, column_name=class_2 ..."
##    '''
##    categorical_column = categorical_column.astype(str)
##    categorical_column = categorical_column.fillna("NULL")
##    ohe = pd.get_dummies(categorical_column, prefix=column_name,prefix_sep='=')
##    return ohe
##def temp_build_ohe_columns(ohe):
##    """
##    ohe is the output of temp_build_ohe
##    
##    """
##    ohe_columns = ohe
##    # TODO: add things to ohe_columns
##    #
##    return ohe_columns
##def add_ohe_columns(df, ohe_columns):   
##    df = add_columns(df,ohe_columns)
##    return df
##def add_ohe_columns_to_df(df, column_name,drop_preohe=False):
##    '''
##    Main function that does temp_build_ohe, temp_build_ohe_columns, and add_ohe_columns,
##    and additional option to drop the original categorical column from df with the arg drop_preohe
##    
##    Args:
##    df (pd.DataFrame): given dataframe
##    column_name (str): the column name of the categorical column that is to be one-hot-encoded
##    drop_preohe (bool): boolean to decide to drop the categorical column from df
##    
##    Returns:
##    pd.DataFrame, the given dataframe with the new one-hot-encoded columns
##
##    Raises:
##    AssertionError: if column_name is not in df   
##    '''
##    assert column_name in list(df.columns)
##    
##    series = df[column_name]
##    ohe = temp_build_ohe(series,column_name)
##    ohe_columns = temp_build_ohe_columns(ohe) 
##    df = add_ohe_columns(df, ohe_columns)
##    if drop_preohe:
##        df.drop(columns=[column_name],inplace=True)
##    return df


## Logging the columns
def get_log_column(loggable_numerical_column):
    '''
    Applies np.log1p (LOG(x+1)) to a loggable numerical column with values >= 0

    Args:
    loggable_numerical_column (pd.Series), column to be log-transformed
    
    Returns:
    pd.Series of the now log-transformed column
    
    Raises:
    AssertionError: if given column's values are not >= 0
    '''
    #assert np.all(loggable_numerical_column >= 0)
    return np.log1p(loggable_numerical_column)
def add_log_column(df, logged_column,logged_column_name):
    '''
    Applies np.log1p (LOG(x+1)) to a numerical column with values >= 0
    And adds it as a column to the df with the new column name LOG(old_column_name)

    Args:
    df (pd.DataFrame): given dataframe
    logged_column (pd.Series), already log-transformed column to be inserted into df
    logged_column_name (str): the column name of the new logged column
    
    Returns:
    pd.DataFrame
    '''
    df[logged_column_name] = logged_column
    return df
def add_log_column_to_df(df, column_name,logged_column_name=None,drop_prelogged=False):
    '''
    Main function that does both get_log_column and add_log_column,
    and additional option to drop the original pre-logged column from df with the arg drop_prelogged
    
    Args:
    df (pd.DataFrame): given dataframe
    logged_column (str): the column name of the pre-logged column,
    logged_column_name (str): the column name of the planned logged column, optional with default as None. If None, goes by the LOG(column_name) format
    drop_prelogged (bool): boolean to decide to drop the pre-logged column from df
    
    Returns:
    pd.DataFrame, the given dataframe with the new log transformed column

    Raises:
    AssertionError: if column_name is not in df  
    '''
    assert column_name in list(df.columns)
    if logged_column_name is None:
       logged_column_name = "LOG({0})".format(column_name)
    column = df[column_name]
    logged_column = get_log_column(column)
    df = add_log_column(df, logged_column,logged_column_name)
    if drop_prelogged:
        df.drop(columns=[column_name],inplace=True)
    return df

## Converting raw date column to properly cleaned pandas datetime column
def get_pd_datetime_column(date_column):
    
    return pd.to_datetime(date_column)
def convert_pd_datetime_column(df, date_column_name):
    date_column = df[date_column_name] 
    df[date_column_name] = get_pd_datetime_column(date_column)
    return df
## Converting boolean columns to numerical 1 or 0 
def get_boolean_to_num_column(boolean_column):
    return (boolean_column).astype(float)
def convert_boolean_to_num_column(df, boolean_column_name):
    boolean_column = df[boolean_column_name] 
    df[boolean_column_name] = get_boolean_to_num_column(boolean_column)
    return df
## Inverse columns
##def get_inverse_column(column,zero_div_prevention=0.0001):
##    assert zero_div_prevention > 0
##    return 1/(column + zero_div_prevention)
##def add_inverse_column(df,column_name,zero_div_prevention=0.0001):
##    
##    assert column_name in list(df.columns)
##    column = df[column_name]
##    assert np.all(column >= 0) # the column to be inverted cannot have any negative numbers
##    inverse_column = get_inverse_column(column,zero_div_prevention)
##    df["1/"+column_name] = inverse_column
##    return df



def time_subtraction_weekends_removal(df, time_end_column_name, time_start_column_name):
    start_day_column = pd.to_datetime(df[time_start_column_name]).dt.date
    end_day_column = pd.to_datetime(df[time_end_column_name]).dt.date
    num_business_days_column = np.busday_count( start_day_column , end_day_column)
    num_seconds_in_a_day = 60*60*24
    
    time_between_end_and_start_column = (
        pd.to_datetime(df[time_end_column_name]) - pd.to_datetime(df[time_start_column_name])
    )  
    days_between_end_and_start_column = (
     time_between_end_and_start_column / np.timedelta64(1, 'd')
    )
    weekends_between_end_and_start_column = days_between_end_and_start_column - num_business_days_column
    weekends_between_end_and_start_column_in_seconds = weekends_between_end_and_start_column * num_seconds_in_a_day
    return weekends_between_end_and_start_column_in_seconds
def get_time_subtraction(df, time_end_column_name, time_start_column_name):
    
    time_between_end_and_start_column = (
        pd.to_datetime(df[time_end_column_name]) - pd.to_datetime(df[time_start_column_name])
    )  
    seconds_between_end_and_start_column = (
     time_between_end_and_start_column / np.timedelta64(1, 's')
    )
    weekend_removal = 0
    # TODO: debug time_subtraction_weekends_removal's datetime y/m/d formatting to make this error stop happening
    try:
        weekend_removal = time_subtraction_weekends_removal(df, time_end_column_name, time_start_column_name)
    except:
        print("Error: could not get time_subtraction_weekends_removal to work; weekends will be kept in")
    seconds_between_end_and_start_column -= weekend_removal
    return seconds_between_end_and_start_column    

def add_time_between_2_events_column(df,
                                     column_name="TIME_BETWEEN_ORDER_AND_DEADLINE",
                                     start_time_column_name="ORDER_DATETIME_PST",
                                     end_time_column_name="PICKUP_DEADLINE_PST",
                                     ):
    '''
    Adds a duration between 2 events column to df. Sometimes resulting rows will have "negative duration", which will be treated as errors and removed

    Args:
    df (pd.DataFrame): the given dataframe
    column_name (str): the new column name to give to the duration column
    start_time_column_name (str): the starting events column name
    end_time_column_name (str): the end events column name
    
    Results:
    pd.DataFrame, the given dataframe with a new duration column
    
    Raises:
    AssertionError: if necessary column names are not in the given dataframe
    '''
    assert start_time_column_name in list(df.columns)
    assert end_time_column_name in list(df.columns)

    seconds_between_2_events_column = get_time_subtraction(df, end_time_column_name,start_time_column_name)
    df[column_name] = seconds_between_2_events_column
    error_rows = df[df[column_name] <0] # rows with negative time
    print("removing {0} negative duration rows".format(error_rows.shape[0]))
    df = df[df[column_name] >=0]
    return df
def add_distance_column(df,
                        column_name="APPROXIMATE_DRIVING_ROUTE_MILEAGE",
                        start_x_coord_column_name = "X_COORD_ORIG",
                        start_y_coord_column_name = "Y_COORD_ORIG",
                        end_x_coord_column_name = "X_COORD_DEST",
                        end_y_coord_column_name = "Y_COORD_DEST",
                       ):
    '''
    Adds a euclidean distance between 2 points column to df. Namely, (but not limited to) the origin and destination zipcode centroids

    Args:
    df (pd.DataFrame): the given dataframe
    column_name (str): the new column name to give to the distance column
    start_x_coord_column_name (str): the starting point's x coord column name
    start_y_coord_column_name (str): the starting point's y coord column name
    end_x_coord_column_name (str): the endpoint's x coord column name
    end_y_coord_column_name (str): the endpoint's y coord column name
    
    Results:
    pd.DataFrame, the given dataframe with a new distance column
    
    Raises:
    AssertionError: if necessary column names are not in the given dataframe
    '''
    assert start_x_coord_column_name in list(df.columns)
    assert start_y_coord_column_name in list(df.columns)
    assert end_x_coord_column_name in list(df.columns)
    assert end_y_coord_column_name in list(df.columns)

    distance_column = (
        (df[end_x_coord_column_name] - df[start_x_coord_column_name])**2 + \
        (df[end_y_coord_column_name] - df[start_y_coord_column_name])**2
    )
    df[column_name] = distance_column
    #df = add_inverse_column(df,column_name)
    return df
def add_distance_over_time_column(df,
                                  column_name="DISTANCE_OVER_ORDER_TIME",
                                  distance_column_name="APPROXIMATE_DRIVING_ROUTE_MILEAGE",
                                  time_column_name="TIME_BETWEEN_ORDER_AND_DEADLINE",
                                  zero_div_prevention=0.0001,
                                  log_normalized=True,
                                       ):
    '''
    Adds a distance over time column to df.

    Args:
    df (pd.DataFrame): the given dataframe
    column_name (str): the new column name to give to the planned distance over time column
    distance_column_name (str): the distance column name
    time_column_name (str): the time duration column name
    zero_div_prevention (float): zero division preventer. must be between 0 and 1
    log_normalized (bool): will log-transforms both the distance and time columns before dividing them with one another
    
    Results:
    pd.DataFrame, the given dataframe with a new distance over time column

    Raises:
    AssertionError: if necessary column names are not in the given dataframe
    '''
    assert distance_column_name in list(df.columns)
    assert time_column_name in list(df.columns)

    distance_column = df[distance_column_name]
    time_column = df[time_column_name]

    if log_normalized:
        distance_column = np.log1p(distance_column)
        time_column = np.log1p(time_column)
    distance_over_time_column = (distance_column / (time_column + zero_div_prevention))
    df[column_name] = distance_over_time_column
    return df
def add_month_column(df, date_column_name="PICKUP_DEADLINE_PST", column_name="MONTH"):
    '''
    Adds a month integer column to df. January = 1, December = 12
    This new column is planned to be a categorical column to be one-hot-encoded

    Args:
    df (pd.DataFrame): the given dataframe
    date_column_name (str): the column name of the date column
    column_name (str): the new column name to give to the planned month column
        
    Results:
    pd.DataFrame, the given dataframe with a new month column

    Raises:
    AssertionError: if date_column_name is not in the given dataframe
    '''
    assert date_column_name in list(df.columns)
    month_column = pd.to_datetime(df[date_column_name]).dt.month
    df[column_name] = month_column
    return df
def add_weekday_column(df,date_column_name="PICKUP_DEADLINE_PST",column_name="DAY"):
    '''
    Adds a weekday integer column to df. Monday = 0, Sunday = 6
    This new column is planned to be a categorical column to be one-hot-encoded

    Args:
    df (pd.DataFrame): the given dataframe
    date_column_name (str): the column name of the date column
    column_name (str): the new column name to give to the planned day column
        
    Results:
    pd.DataFrame, the given dataframe with a new weekday column

    Raises:
    AssertionError: if date_column_name is not in the given dataframe
    '''
    assert date_column_name in list(df.columns)
    day_column = pd.to_datetime(df[date_column_name]).dt.weekday
    df[column_name] = day_column
    return df
def add_hour_column(df, date_column_name="PICKUP_DEADLINE_PST", column_name="HOUR"):
    '''
    Adds a hour integer column to df. 24 hours
    This new column is planned to be a categorical column to be one-hot-encoded

    Args:
    df (pd.DataFrame): the given dataframe
    date_column_name (str): the column name of the date column
    column_name (str): the new column name to give to the planned hour column
        
    Results:
    pd.DataFrame, the given dataframe with a new hour column

    Raises:
    AssertionError: if date_column_name is not in the given dataframe
    '''
    assert date_column_name in list(df.columns)
    hour_column = pd.to_datetime(df[date_column_name]).dt.hour
    df[column_name] = hour_column
    return df






# helper functions to add columns specifically to oa offers

def add_offer_type_is_pooled_column(oa_offers,column_name="OFFER_TYPE_IS_POOLED"):
    '''
    Adds if offer type is pooled
    '''
    assert "OFFER_TYPE" in list(oa_offers.columns)
    oa_offers[column_name] = (oa_offers["OFFER_TYPE"]=="pool").astype(float)
    return oa_offers


# helper functions to add columns specifically to oa (the join of oa offers and oa orders)
def add_offer_order_amount_column(oa,column_name='ORDER_OFFER_AMOUNT',reference_number_column_name="REFERENCE_NUMBERS"):
    '''
    Adds column denoting number of offers per order ("reference numbers") to oa
    
    '''
    assert reference_number_column_name in list(oa.columns)
    count_groupby = oa.groupby([reference_number_column_name]).size().reset_index(name=column_name)
    oa = oa.merge(count_groupby,on=[reference_number_column_name])
    return oa

def add_lead_time_column(oa, oa_offers, column_name = "LEAD_TIME", reference_number_column_name="REFERENCE_NUMBERS"):
    '''
    Adds a lead time column to the oa dataframe.
    The lead time percentage or "patience" column is the percentage of the duration between the
    chosen offer's proposal and the order's creation, divided by the duration between pickup deadline and order's creation

    This column should NOT be used as a feature for the final model, but rather a weight column for sample_weights
    which will punish orders where the selected offer was selected overly-early or impatiently

    Args:
    oa (pd.DataFrame): the oa data
    oa_offers (pd.DataFrame): the offers data
    column_name (str): column name to give to the new lead-time column name
    reference_number_column_name (str): the column that joins oa and oa_offerss
    
    Returns:
    pd.DataFrame, the oa with the new lead time column added
    
    Raises:
    AssertionError: if necessary column names are not in the given dataframe
    '''
    # adds a new column called LEAD_TIME to oa, which will be used to weight by lead time percentage for the model's training.
    assert "LOAD_DELIVERED_FROM_OFFER" in list(oa_offers.columns)
    assert reference_number_column_name in list(oa_offers.columns)
    assert "CREATED_ON_HQ" in list(oa_offers.columns)
    assert (("TIME_BETWEEN_ORDER_AND_DEADLINE" in list(oa.columns)) or ("LOG(TIME_BETWEEN_ORDER_AND_DEADLINE)" in list(oa.columns)))
    assert reference_number_column_name in list(oa.columns)
    assert "ORDER_DATETIME_PST" in list(oa.columns)
    # get only the offers that were chosen
    oa_offers_delivered = oa_offers[oa_offers["LOAD_DELIVERED_FROM_OFFER"]==1]

    # among these chosen offers, get the dates when they were offered by a carrier (CREATED_ON_HQ)
    oa_offers_delivered = oa_offers_delivered[[reference_number_column_name,"CREATED_ON_HQ"]]

    # merge oa with the oa_offers_delivered
    # this join should be 1-to-1 (i.e. no duplicate orders are created): recall that oa's row schema is orders (i.e. each row is a unique order)
    # while oa_offers_delivered should be also now of a row schema the moment we filtered only for the chosen offers
    # because each order can only have 1 chosen offer in the end
    
    oa = oa.merge(oa_offers_delivered, on=[reference_number_column_name])
    time_between_order_and_offer_column = get_time_subtraction(oa, "CREATED_ON_HQ","ORDER_DATETIME_PST")
    # get the time difference or duration between when the order was made and when the chosen offer was proposed
    oa.drop(columns=["CREATED_ON_HQ"],inplace=True)

    # depending on which version of data-cleaning was used, the divisor column or time_between_order_and_deadline_column is either logged or not
    if "TIME_BETWEEN_ORDER_AND_DEADLINE" in list(oa.columns):
        time_between_order_and_deadline_column = oa["TIME_BETWEEN_ORDER_AND_DEADLINE"]
    else:
        if "LOG(TIME_BETWEEN_ORDER_AND_DEADLINE)" in list(oa.columns):
            # if logged, the logging must be undone
            time_between_order_and_deadline_column = np.expm1(oa["LOG(TIME_BETWEEN_ORDER_AND_DEADLINE)"])
    
    zero_div_prevention = 0.0001
    lead_time_percentage_column =  time_between_order_and_offer_column/(time_between_order_and_deadline_column + zero_div_prevention)
    oa[column_name] = lead_time_percentage_column

    # final data cleaning of error rows
    oa[column_name] = oa[[column_name]].replace([np.inf, -np.inf], np.nan).values
    oa = oa.dropna(subset=[column_name])
    
    oa = oa[((oa[column_name] >= 0) & (oa[column_name] <= 1))] # since lead_time is a percentage, any rows < 0 or > 1 are errors and must be dropped
    return oa




def temp_build_metro_cluster(oa,
                            x_coord_column_name = "X_COORD_DEST",
                            y_coord_column_name = "Y_COORD_DEST",
                            value_column_name="ORDER_OFFER_AMOUNT",
                            centroid_as_metro=False,
                            group_amount=20):
    #print("Do not drop ORDER_OFFER_AMOUNT column before clustering the metro areas")
    assert x_coord_column_name in list(oa.columns)
    assert y_coord_column_name in list(oa.columns)
    assert value_column_name in list(oa.columns)

    zipcode_density = (oa.groupby([x_coord_column_name, y_coord_column_name],as_index=False).agg({value_column_name:np.sum}))
    if value_column_name == "ORDER_OFFER_AMOUNT":
        density_adjuster = 10
        zipcode_density[value_column_name] = np.log1p(zipcode_density[value_column_name])/density_adjuster
    metro_cluster = Metro_Cluster(
        df=zipcode_density,
        x_column_name=x_coord_column_name,
        y_column_name=y_coord_column_name,
        value_column_name=value_column_name,
        cluster_method="kmeans",
        group_amount=group_amount
        )
    metro_cluster.build_cluster_df(centroid_as_metro=centroid_as_metro)
    metro_cluster.build_opt_pivot()
    return metro_cluster
def temp_build_metro_cluster_columns(oa,
                                     metro_cluster,
                                     x_coord_column_name,
                                     y_coord_column_name,
                                     metro_cluster_column_names,
                                     ):
    #print("Do not drop ORDER_OFFER_AMOUNT column before clustering the metro areas")
    assert x_coord_column_name in list(oa.columns)
    assert y_coord_column_name in list(oa.columns)

    metro_cluster_columns = metro_cluster.get_proximities(oa,
                                                x_coord_column_name,
                                                x_coord_column_name,
                                                metro_cluster_column_names)
    return metro_cluster_columns
def add_metro_cluster_columns(oa, metro_cluster_columns):
    oa = add_columns(oa,metro_cluster_columns)
    return oa
def add_metro_cluster_columns_to_df(oa):
    '''
    Clusters all the zipcodes into N groups, then one-hot-encodes the order's orig and dest zipcodes
    into those groups (for a total of 2*N columns). These columns aren't really one-hot-encoded like binary,
    but are like softmaxes for how close a zipcode is to a centroid of a cluster.
    For example, let's say all the zipcodes are clustered into 3 groups A, B, C 
    if a given order had its orig zipcode very close to B, and its dest zipcode was equidistant from all groups,
    its result would like [0.1, 0.8, 0.1] + [0.33, 0.33, 0.33]
    or conjoined as [0.1, 0.8, 0.1, 0.33, 0.33, 0.33], with columns 
    ["ORIG_GROUP0", "ORIG_GROUP1","ORIG_GROUP2","DEST_GROUP0", "DEST_GROUP1","DEST_GROUP2"]
    '''
    assert "X_COORD_DEST" in list(oa.columns)
    assert "X_COORD_ORIG" in list(oa.columns)
    assert "Y_COORD_DEST" in list(oa.columns)
    assert "Y_COORD_ORIG" in list(oa.columns)
    metro_cluster = temp_build_metro_cluster(oa)
    orig_proximity_column_names = ["ORIG_"+metro_cluster.group_column_name+"="+str(i) for i in range(metro_cluster.group_amount)]
    orig_metro_cluster_columns = temp_build_metro_cluster_columns(oa, metro_cluster,"X_COORD_ORIG","Y_COORD_ORIG",orig_proximity_column_names)
    oa = add_metro_cluster_columns(oa, orig_metro_cluster_columns)

    dest_proximity_column_names = ["DEST_"+metro_cluster.group_column_name+"="+str(i) for i in range(metro_cluster.group_amount)]
    dest_metro_cluster_columns = temp_build_metro_cluster_columns(oa, metro_cluster,"X_COORD_DEST","Y_COORD_DEST",dest_proximity_column_names)
    oa = add_metro_cluster_columns(oa, dest_metro_cluster_columns)
    return oa



##def temp_build_path_groupby(oa):
##    assert "DESTINATION_3DIGIT_ZIP" in list(oa.columns)
##    assert "ORIGIN_3DIGIT_ZIP" in list(oa.columns)
##    assert 'LOG(RATE_USD)' in list(oa.columns)
##    assert 'REFERENCE_NUMBERS' in list(oa.columns)
##    reference_numbers_column_name = "REFERENCE_NUMBERS"
##    
##    
##    temp_oa_collapsed_orders = oa.copy()
##
##    temp_aggdict = dict()
##    temp_oa_collapsed_orders["PATH_AVG_LOG(RATE_USD)"] = temp_oa_collapsed_orders["LOG(RATE_USD)"]
##    temp_aggdict["PATH_AVG_LOG(RATE_USD)"] = np.mean
##
##    temp_oa_collapsed_orders["PATH_ORDER_AMOUNT"] = temp_oa_collapsed_orders["REFERENCE_NUMBERS"]
##    temp_aggdict["PATH_ORDER_AMOUNT"] = "count"
##
##    if "LOG(ORDER_OFFER_AMOUNT)" in list(temp_oa_collapsed_orders.columns):
##        temp_oa_collapsed_orders["PATH_AVG_LOG(ORDER_OFFER_AMOUNT)"] = temp_oa_collapsed_orders["LOG(ORDER_OFFER_AMOUNT)"]
##        temp_aggdict["PATH_AVG_LOG(ORDER_OFFER_AMOUNT)"] = np.mean
##    else:
##        pass
##    if "ORDER_OFFER_AMOUNT" in list(temp_oa_collapsed_orders.columns):
##        temp_oa_collapsed_orders["PATH_AVG_ORDER_OFFER_AMOUNT"] = temp_oa_collapsed_orders["ORDER_OFFER_AMOUNT"]
##        temp_aggdict["PATH_AVG_ORDER_OFFER_AMOUNT"] = np.mean
##    else:
##        pass
##    if "LOG(OPER_COUNT)" in list(temp_oa_collapsed_orders.columns):
##        temp_oa_collapsed_orders["PATH_AVG_LOG(OPER_COUNT)"] = temp_oa_collapsed_orders["LOG(OPER_COUNT)"]
##        temp_aggdict["PATH_AVG_LOG(OPER_COUNT)"] = np.mean
##    else:
##        pass
##    if "LOG(TEMPERATURE)" in list(temp_oa_collapsed_orders.columns):
##        temp_oa_collapsed_orders["PATH_AVG_LOG(TEMPERATURE)"] = temp_oa_collapsed_orders["LOG(TEMPERATURE)"]
##        temp_aggdict["PATH_AVG_LOG(TEMPERATURE)"] = np.mean
##    else:
##        pass
##
##    print(temp_aggdict)
##    
##
##    temp_groupby = (
##        temp_oa_collapsed_orders
##        .groupby([#"PATH",
##                  "X_COORD_ORIG","Y_COORD_ORIG",
##                  "X_COORD_DEST","Y_COORD_DEST"],as_index=False)
##        .agg(temp_aggdict)
##    )
##    return temp_groupby
##
##def add_path_columns_to_df(oa):
##    '''
##    Adds path aggregation columns
##    This is just for visualization purposes and none of these columns should actually be used for the model
##    '''
##
##    assert 'REFERENCE_NUMBERS' in list(oa.columns)
##    temp_groupby =  temp_build_path_groupby(oa)
##    oa = oa.merge(temp_groupby,on=[
##                  "X_COORD_ORIG","Y_COORD_ORIG",
##                  "X_COORD_DEST","Y_COORD_DEST",
##                 ])
##    return oa


def view_pca(X,y):
    pca = PCA(2)
    Z = pca.fit_transform(X)
    fig, axs = plt.subplots()
    axs.scatter(Z[:,0], Z[:,1], c=y,s=1, alpha=0.1)
    
    return pca


# Because of a lack of control with Sklearn Pipelines, custom classes for building the models were made.
##class Temp_Order_Prediction_Model_Builder:
##    def __init__(
##        self,
##        df,
##        target_column_name,
##        weight_column_name=None,
##    ):
##        '''
##        
##        '''
##        assert target_column_name in list(df.columns)
##
##
##        
##        self.df = df
##        self.target_column_name = target_column_name
##        self.weight_column_name = weight_column_name
##        #if (weight_column_name is None)==False:
##        #    assert weight_column_name in list(df.columns)
##        self.X = self.df.drop(columns=[self.target_column_name])
##        self.y = self.df[self.target_column_name]
##    def _verify_weight(self):
##        # checks if this model will be weighing by lead time.
##        if self.weight_column_name is None:
##            return False
##        else:
##            return True
##    def get_X(self):
##        # gets X (the input df of the model which lacks the target column)
##        # if we were not using weighting by lead-time, calling self.get_X() and self.X are indistinguishable
##        # but if using weighting by leadtime, self.get_X() will return X without the leadtime column
##        X = self.X
##        if self._verify_weight():
##            return X.drop(columns=[self.weight_column_name],inplace=False)
##        else:
##            return X
##    def _record_post_split_weight(self):
##        # if the weight_column_name is being used, then it must be dropped from self.X_train and recorded
##        if self._verify_weight():
##            print("sample_weights will be used")
##            weight_column_name = self.weight_column_name
##            X_train_weight_column = self.X_train[weight_column_name]
##            self.X_train = self.X_train.drop(columns=[weight_column_name])
##            self.X_test = self.X_test.drop(columns=[weight_column_name])
##            self.X_train_weight_column = X_train_weight_column
##            #self.temp_weight_column = X_train_weight_column
##            print("")
##        else:
##            pass
##    
##    def transform_y(self, y=None, inplace=False):
##        if y is None: 
##            y = self.y;
##        # y = np.log1p(y)
##        # y = np.sqrt(y)
##
##        if inplace==True: 
##            self.y = y        
##        return y
##    def transform_X(self, X=None, top_n = None,inplace=False):
##        if X is None: 
##            X = self.X;
##        if top_n is None:
##            top_n = self.get_X().shape[1]
##        
##        #top_n = 30
##        target_column_name = self.target_column_name
##        df = self.df
##        assert target_column_name in list(df.columns)
##
##        if self._verify_weight():
##            weight_column_name = self.weight_column_name
##            weight_column = df[weight_column_name]
##            df.drop(columns=[weight_column_name], inplace=True)
##        
##        top_n_correlated_selected_column_names = (
##            df.corr()[target_column_name]
##            .abs()
##            .sort_values(ascending=False)
##            .head(top_n+1).index[1:])
##        
##        self.top_n_correlated_selected_column_names = top_n_correlated_selected_column_names
##        X = X[top_n_correlated_selected_column_names]
##        
##        if self._verify_weight():
##            weight_column_name = self.weight_column_name
##            X[weight_column_name] = weight_column
##        
##        if inplace==True: 
##            self.X = X
##        return X
##    def set_split(self, X=None, y=None, stratify=None,train_test_indexer=None):
##        if X is None:
##            X = self.X
##        if y is None:
##            y = self.y
##        # train_test_indexer is a boolean array, 1 = train, 0 = test
##        if train_test_indexer is None:
##            stratify = stratify
##            np.random.seed(1)
##            X_train, X_test, y_train, y_test = train_test_split(X,y,stratify)
##        else:
##            X_train = X.loc[train_test_indexer == 1]
##            y_train = y.loc[train_test_indexer == 1]
##            X_test = X.loc[train_test_indexer == 0]
##            y_test = y.loc[train_test_indexer == 0]
##        self.X_train = X_train
##        self.X_test = X_test
##        self.y_train = y_train
##        self.y_test = y_test
##        self._record_post_split_weight()
##        
##    def train_model(self, model, X_train=None, y_train=None):
##        if X_train is None:
##            X_train = self.X_train
##        if y_train is None:
##            y_train = self.y_train
##        
##        if self._verify_weight():
##            
##            weight_column = self.X_train_weight_column
##            fit_params = {'sample_weight': weight_column}
##            try: 
##                print("Cross Val Scores:",cross_val_score(model, X_train, y_train, fit_params=fit_params))
##            except:
##                print("Cross Val Scores skipped")
##            model.fit(X_train, y_train,sample_weight=weight_column)
##        else:
##            try: 
##                print("Cross Val Scores:",cross_val_score(model, X_train, y_train))
##            except:
##                print("Cross Val Scores skipped")
##            model.fit(X_train, y_train)
##        self.model = model
##        return model
##    def test_model(self, model=None,X_test=None,y_test=None):
##        if model is None:
##            model = self.model
##        if X_test is None:
##            X_test = self.X_test
##        if y_test is None:
##            y_test = self.y_test
##        predictions = model.predict(X_test)
##        return predictions
##
##class Temp_Order_Classification_Model_Builder (Temp_Order_Prediction_Model_Builder):
##    def __init__(
##        self,
##        df,
##        target_column_name,
##        class_amount = 2,
##        weight_column_name=None,
##    ):
##        super(Temp_Order_Classification_Model_Builder, self).__init__(df, target_column_name,weight_column_name)
##        self.class_amount = class_amount
##    def _split_as_n_ordinal_levels(self, y,class_amount=None):
##        if class_amount is None:
##            class_amount = 2 #4
##        percentiles = [np.percentile(y,x) for x in np.arange(0,100,100/class_amount)[1:]]
##        self.percentiles = percentiles
##        ordinals = np.zeros(y.shape[0])
##        for percentile in percentiles:
##            ordinals += (y>=percentile).astype(float)
##        y = ordinals
##        return y
##    def set_split(self, X=None, y=None, stratify=None,train_test_indexer=None):
##        if X is None:
##            X = self.X
##        if y is None:
##            y = self.y
##        if train_test_indexer is None:
##            if stratify is None:
##                stratify = self.y
##            np.random.seed(1)
##            X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=stratify)
##        else:
##            X_train = X.loc[train_test_indexer == 1]
##            y_train = y.loc[train_test_indexer == 1]
##            X_test = X.loc[train_test_indexer == 0]
##            y_test = y.loc[train_test_indexer == 0]
##        self.X_train = X_train
##        self.X_test = X_test
##        self.y_train = y_train
##        self.y_test = y_test
##
##        #print(self.X_train.shape,self.X_test.shape,self.y_train.shape,self.y_test.shape)
##        
##        self._record_post_split_weight()
##    def transform_y(self, y=None, inplace=False
##                   ):
##        if y is None: 
##            y = self.y;
##        y = self._split_as_n_ordinal_levels(y, class_amount=self.class_amount)
##        if inplace==True: 
##            self.y = y
##        return y
##
##        
##            
##            
##
##    def train_model(self, model, X_train=None, y_train=None):
##        if X_train is None:
##            X_train = self.X_train
##        if y_train is None:
##            y_train = self.y_train
##
##        if self._verify_weight():
##            weight_column = self.X_train_weight_column
##            fit_params = {'sample_weight': weight_column}            
##            if self.class_amount == 2:
##                print("Cross Val Scores:",cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=4,shuffle=True), fit_params=fit_params,scoring="roc_auc"))
##            else:
##                print("Cross Val Scores:",cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=4,shuffle=True), fit_params=fit_params))
##            model.fit(X_train, y_train,sample_weight=weight_column)   
##        else:
##            if self.class_amount == 2:
##                print("Cross Val Scores:",cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=4,shuffle=True),scoring="roc_auc"))
##            else:
##                print("Cross Val Scores:",cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=4,shuffle=True)  ))
##            model.fit(X_train, y_train)
##        self.model = model
##        return model
##    def eval_model(self, predictions,y_test=None):
##        if y_test is None: 
##            y_test = self.y_test
##        print("Confusion Matrix:\n", confusion_matrix(y_test,predictions,normalize="true"))
##        if self.class_amount==2:
##            print("ROC AUC Score", roc_auc_score(y_test,predictions))
##            pass
##
##
##    def oversample_class_imbalance(self):
##        '''Optional function to oversample the minority class if there is class_imbalance; Performed between set_split and train_model'''
##        #temp_X_train = self.X_train
##        if self._verify_weight():
##            # add  column to x train to make things easier
##            self.X_train[self.weight_column_name] = self.X_train_weight_column
##        else:
##            pass
##        classes = list(set(self.y_train.values.tolist()))
##        class_i_X_train_dfs = [self.X_train[self.y_train==c] for c in classes]
##        class_i_X_train_amounts = [df.shape[0] for df in class_i_X_train_dfs]
##        max_class_i_X_train_amount = max(class_i_X_train_amounts)
##        for i in range(len(classes)):
##            class_value = classes[i]
##            class_i_X_train_df = class_i_X_train_dfs[i]
##            class_i_X_train_amount = class_i_X_train_amounts[i]
##            class_amount_difference = max_class_i_X_train_amount - class_i_X_train_amount
##            assert class_amount_difference >= 0
##            if class_amount_difference > 0:
##                # class i is less than class max
##                extra_class_i_X_train_df = class_i_X_train_df.sample(n=class_amount_difference,replace=True)
##                self.X_train = pd.concat([self.X_train,extra_class_i_X_train_df],axis=0)
##                self.y_train = pd.Series((self.y_train.values.tolist() + (class_value*np.ones(class_amount_difference)).tolist()),
##                                        name=self.target_column_name)
##            else:
##                # class i is equal to class max
##                pass
##        if self._verify_weight():
##            new_X_train_weight_column = self.X_train[self.weight_column_name]
##            self.X_train.drop(columns=[self.weight_column_name], inplace=True)
##            self.X_train_weight_column = new_X_train_weight_column
##        else:
##            pass
##
##class Temp_Order_Regression_Model_Builder (Temp_Order_Prediction_Model_Builder):
##    def __init__(
##        self,
##        df,
##        target_column_name,
##        weight_column_name=None,
##    ):
##        super(Temp_Order_Regression_Model_Builder, self).__init__(df, target_column_name,weight_column_name)
##    def set_split(self, X=None, y=None, stratify=None,train_test_indexer=None):
##        if X is None:
##            X = self.X
##        if y is None:
##            y = self.y
##        if train_test_indexer is None:
##            np.random.seed(1)
##            X_train, X_test, y_train, y_test = train_test_split(X,y)
##        else:
##            X_train = X.loc[train_test_indexer == 1]
##            y_train = y.loc[train_test_indexer == 1]
##            X_test = X.loc[train_test_indexer == 0]
##            y_test = y.loc[train_test_indexer == 0]
##        self.X_train = X_train
##        self.X_test = X_test
##        self.y_train = y_train
##        self.y_test = y_test
##        #print(self.X_train.shape,self.X_test.shape,self.y_train.shape,self.y_test.shape)
##
##        self._record_post_split_weight()
##    def transform_y(self, y=None, inplace=False,
##                    update_df=False
##):
##        if y is None: 
##            y = self.y;
##        if inplace==True: 
##            self.y = y
##        if update_df==True:
##            self.df[self.target_column_name] = y
##        return y
##    def eval_model(self, predictions,y_test=None):
##        if y_test is None: 
##            y_test = self.y_test
##        print("Corr between Regression Predicted Y & Actual Y:", np.corrcoef(predictions, (y_test))[0][1])



def plot_model_pipeline_feature_importances(model_pipeline, path_folder, plotname):
    # feature importances
    try:
        model_feature_importances = np.abs(model_pipeline.named_steps["model"].coef_.flatten())
    except:
        model_feature_importances = model_pipeline.named_steps["model"].feature_importances_
    #model_feature_importances = model_pipeline.named_steps["model"].feature_importances_
    model_feature_names = model_pipeline.named_steps["preprocessor_pipeline"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(10,5))
    pd.Series(
        index=model_feature_names,
        data=model_feature_importances,
    ).sort_values(ascending=False).head(15)[::-1].plot(kind="barh",ax=ax)
    ax.set_title(plotname.split('.')[0].replace("_"," ").upper())
    fig.savefig(os.path.join(path_folder,'generated_visualizations',plotname),bbox_inches='tight')
    plt.show()
    return fig, ax

