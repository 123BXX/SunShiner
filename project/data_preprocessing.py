import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

NUMLIST = [str(i) for i in range(10)]

"""
Construct a function to extract a stop_id from alighting stop in the traffic dataset

Parameters:
    stop_string: The alighting stop string to be extracted
Return:
    s_num: the stop_id number in stop string
"""
def getNum(stop_string):
    status = False
    s_num = ''

    for i, num in enumerate(stop_string):
        if num == '[':
            status = True

        if status == True:
            if num in NUMLIST:
                s_num += num

        if num == ']':
            status = False

    return s_num

"""
Construct a function to create split the traffic data by the time interval of 15 minutes

Parameters:
    start_date: The start date of the time interval
    end_date: The end date of the time interval
Return:
    time_slices: a time index to group the traffic data into 15 minutes time interval
"""
def date_slice(start_date, end_date):
    time_slices = pd.Series()
    start_date = pd.to_datetime(start_date, format = '%Y/%m/%d %H:%M')
    end_date = pd.to_datetime(end_date, format = '%Y/%m/%d %H:%M')
    time_slices = time_slices.append(pd.Series(start_date), ignore_index = True)

    while end_date > start_date :
        interval = timedelta(minutes = 15)
        start_date = start_date + interval
        time_slices = time_slices.append(pd.Series(start_date), ignore_index = True)

    return time_slices

def casting_date(date):
    date = pd.to_datetime(date, format = '%Y%m%d %H:%M')
    return date

"""
Construct a function to save the database

Parameters:
    dataset: The dataset to be saved
    filename: The filename of dataset
Return:
    time_slices: a time index to group the traffic data into 15 minutes time interval
"""
def save_csv(dataset, filename):
    dataset.to_csv(filename)

"""
Construct a function to read a text file

Parameters:
    filename: The filename of text
Return:
    destination_set: the destination set to be delivered
"""
def read_text_file(filename):
    # extract the stop information from stop destination text
    destination = pd.read_csv('Stop_Destination.txt', sep=',', engine='python', header=None,
                              skiprows=1, names=['stop_id', 'stop_code', 'stop_name', 'stop_desc',
                                                 'stop_lat', 'stop_lon', 'zone_id', 'stop_url',
                                                 'location_type', 'parent_station', 'platform_code'])
    destination = destination.drop(destination[destination['stop_code'].isnull()].index)
    destination_set = destination[['stop_id', 'stop_code', 'stop_name', 'stop_lat', 'stop_lon']]

    return destination_set

"""
Construct a function to read and handle with traffic data

Parameters:
    filename: The filename of traffic data
Return:
    dataframe: The dataframe of traffic dataset to be analyzed
"""
def read_traffic_csv(filename):
    dataframe = pd.read_csv(filename, low_memory=False)
    # fill missing values in Passengers with its median value
    dataframe['Passengers'] = dataframe['Passengers'].fillna(dataframe['Passengers'].median())
    # replace out-of-range value in Passengers with its median value
    dataframe.loc[dataframe['Passengers'] == 0, 'Passengers'] = dataframe['Passengers'].median()
    dataframe = dataframe.drop(dataframe[dataframe['Passengers'] < 0].index)
    # drop the rows with missing values of alighting stop
    dataframe = dataframe.drop(dataframe[dataframe['Alighting Stop'].isnull()].index)
    # transfer the data type of alighting time into date type
    dataframe['Alighting Time'] = pd.to_datetime(dataframe['Alighting Time'], infer_datetime_format=True)
    # extract the stop_id from the alighting stop string
    dataframe['stop_id'] = dataframe['Alighting Stop'].apply(lambda x: getNum(x))
    dataframe = dataframe[['Operations Date', 'Alighting Time', 'Passengers', 'Alighting Stop', 'stop_id']]
    # start_date = casting_date('2019/04/01 06:00')
    # end_date = casting_date('2019/04/01 23:59')
    # dataframe = dataframe[dataframe['Alighting Time'] >= start_date]
    # dataframe = dataframe[dataframe['Alighting Time'] <= end_date]
    return dataframe

dataframe = read_traffic_csv('May 2019 TransactionReport-314828-1.csv')
# extract the stop information from stop destination text
destination_set = read_text_file('Stop_Destination.txt')
result = pd.merge(dataframe, destination_set, on = 'stop_id')



# the date start from 1st April 2019 to 30th April 2019
date_index = date_slice(start_date ='2019/05/01 06:00', end_date = '2019/05/07 23:59')
result['Time Interval'] = pd.cut(result['Alighting Time'], bins = date_index)
aggregate_result = pd.DataFrame(result.groupby(['Time Interval', 'stop_id'])['Passengers'].sum())

#save_csv(aggregate_result, 'Aggregate_Passenger_Flow_May_01-07.csv')
#save_csv(result, 'Merge_Traffic_Destination.csv')
#save_csv(destination_set, 'destination.csv')












