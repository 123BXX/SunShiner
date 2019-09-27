import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

NUMLIST = [str(i) for i in range(10)] # the number str from 1 to 9
TRAINING_TIME_INTERVAL_NUM = 10 # training size for each
INPUT_SIZE = 1  # rnn input size
LR = 0.02  # learning rate

"""
Construct a function to extract hours and minutes from Time Interval str

Parameters:
    time_str: the time string to be extract:
Returns:
    s_num: the expected number of string
"""
def get_time_period(time_str):
    status = False
    s_num = ''

    for i, num in enumerate(time_str):
        if num == ' ':
            status = True

        if status == True:
            if num in NUMLIST:
                s_num += num

        if len(s_num) == 4:
            status = False

    return s_num

"""
Construct a function to save the database

Parameters:
    dataset: The dataset to be saved
    filename: The filename of dataset
"""
def save_csv(dataset, filename):
    dataset.to_csv(filename)

"""
Construct a function to save the database

Parameters:
    date_string: The date string to be converted
Returns:
    date: The date in format of datetime
"""
def casting_date(date_string):
    date = pd.to_datetime(date_string, format = '%Y%m%d %H:%M')
    return date

"""
Construct a function to extract passenger flow and corresponding date from a stop

Parameters:
    dataframe: The traffic dataframe to be extract
    stop_id: The stop to be analyzed
Return:
    stop: the customized data only contains time interval and passengers
"""
def create_stop_data(stop_id, dataframe):
    stop = dataframe[dataframe['stop_id'] == stop_id]
    stop = stop[['Time Interval', 'Passengers']]
    stop['Time Interval'] = stop['Time Interval'].apply(lambda x: get_time_period(x))
    stop = stop.astype('float32')
    return stop


"""
Construct a function to split the raw data set into input dataset
and output dataset.

Parameters:
    raw_data: The dataset to be processed
    time_interval_num: The input size of dataset for each time step
"""
def create_dataset(raw_data, time_interval_num):
    input_dataset, compared_dataset = [], []
    for i in range(len(raw_data) - time_interval_num):
        input_ele = raw_data[i:(i + time_interval_num)]
        input_dataset.append(input_ele)
        compared_dataset.append(raw_data[i + time_interval_num])
    return (np.array(input_dataset), np.array(compared_dataset))

"""
Construct a function to to normalize data from 0 to 1

Parameters:
    dataset: The dataset to be normalized
Returns:
    dataset: The normalized data
"""
def normalize_dataset(dataset):
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    dataset = (dataset - min_value) / (max_value - min_value)
    return dataset

"""
Construct a function to to store the net

Parameters:
    network: The network to be saved
    name: The file name of the network
"""
def save_network(network, name):
    torch.save(network, name)

def restore_net():
    model = torch.load('net.pkl')
    return model

"""
Construct an LSTM module class to make passenger flow prediction 
"""
class LSTM_REG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(LSTM_REG, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape  # (seq, batch, hidden)
        x = x.view(s * b, h)  # 转化为线性层的输入方式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

# read the processed data from passenger flow file and destination file
dataframe = pd.read_csv('./Aggregate_Passenger_Flow_April_01-07.csv')
dataframe = dataframe.dropna()
destination = pd.read_csv('./Destination.csv')
dataframe['stop_id'] = dataframe['stop_id'].astype(str)

#combine the passenger flow file and destination file
combined_data = pd.merge(destination, dataframe, on = 'stop_id')

# extract the dataset of UQ lake on April 1st, 2019
stop_UQ = create_stop_data('1882', dataframe)
stop_UQ = stop_UQ.astype('float32')
train_set = np.array(stop_UQ['Passengers'].values)
train_set = normalize_dataset(train_set)
input_dataset, compared_dataset = create_dataset(train_set, TRAINING_TIME_INTERVAL_NUM)

train_input = input_dataset.reshape(-1, 1, TRAINING_TIME_INTERVAL_NUM)
train_compared = compared_dataset.reshape(-1, 1, 1)
train_input = torch.from_numpy(train_input)
train_compared = torch.from_numpy(train_compared)

# model = LSTM_REG(10, 64)
model = restore_net()
loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# for i in range(500):
#     out = model(train_input)
#     loss = loss_fun(out, train_compared)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     if (i + 1) % 100 == 0:
#         print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))
#
# torch.save(model, 'net.pkl')
# model = model.eval()


input_dataset = input_dataset.reshape(-1, 1, TRAINING_TIME_INTERVAL_NUM)
input_dataset = torch.from_numpy(input_dataset)
prediction = model(input_dataset)
prediction = prediction.view(-1).data.numpy()
prediction = np.concatenate((np.zeros(TRAINING_TIME_INTERVAL_NUM), prediction))
assert len(prediction) == len(train_set)

plt.plot(prediction.flatten(), 'b', label = 'LSTM Predicted Passenger Flow')
plt.plot(train_set.flatten(), 'r', label = 'Real Passenger Flow')
plt.xlabel('Time Interval : 15 minutes / Duration: April 1st to May 7th')
plt.ylabel('Passenger Number')
plt.title('Passenger Flow for UQ Lake Station [1882] during April 1st and 7th, 2019')
#plt.text(0.5, 0.95, 'Loss=%.4f' % loss.data, fontdict = {'size':10, 'color':'red'})
plt.legend(loc='best')
plt.savefig('result.png', format='png', dpi=200)
plt.show()



# # plot of the stop passenger from April 2019
# #plt.figure('Passenger Flow for UQ Lake Station [1882] during April 2019')
# plt.figure('Passenger Flow for Boggo Road station [10795] during April 1st 2019')
# x_list = stop_UQ.index
# y_list = stop_UQ['Passengers']
# # plt.scatter(x_list, y_list)
# ax = plt.gca()
# ax.plot(x_list, y_list, color='b', linewidth=1, alpha=0.6)
#
# ax.set_ylabel('Passenger Number')
# #ax.set_xlabel('Time Interval : 15 minutes / Duration: April 1st to April 30th')
# ax.set_xlabel('Time Interval : 15 minutes / Duration: April 1st 06:00am to 23:59pm')
# #ax.set_xlabel('Time Interval : 15 minutes / Duration: April 1st to April 7th')
# #ax.set_title('Passenger Flow for UQ Lake Station [1882] during April, 2019')
# #ax.set_title('Passenger Flow for UQ Lake Station [1882] during April, 2019')
# ax.set_title('Passenger Flow for Boggo Road station [10795] during April 1st, 2019')
# #ax.set_title('Passenger Flow for Boggo Road station [10795] during April 1st to April 7th, 2019')
# #ax.set_title('Passenger Flow for Boggo Road station [10795] during April 1st to April 30th, 2019')
# plt.show()

