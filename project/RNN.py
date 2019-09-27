import torch
from torch import nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

NUMLIST = [str(i) for i in range(10)]
TRAINING_TIME_INTERVAL_NUM = 10
INPUT_SIZE = 10  # rnn input size
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
def casting_date(date):
    date = pd.to_datetime(date, format = '%Y%m%d %H:%M')
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
Construct an RNN module class to make passenger flow prediction 
"""
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size = INPUT_SIZE,
            num_layers = 1,
            hidden_size = 32,
            batch_first = True
        )
        self.out = nn.Linear(64, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim = 1), h_state

# read the processed data from passenger flow file and destination file
dataframe = pd.read_csv('./Processed_Result/Aggregate_Passenger_Flow.csv')
dataframe = dataframe.dropna()
destination = pd.read_csv('./Processed_Result/Destination.csv')
dataframe['stop_id'] = dataframe['stop_id'].astype(str)

# combine the passenger flow file and destination file
combined_data = pd.merge(destination, dataframe, on = 'stop_id')

# extract the dataset of UQ lake on April 1st, 2019
stop_UQ = create_stop_data('1882', dataframe)
train_set = np.array(stop_UQ['Passengers'].values)
train_set = normalize_dataset(train_set)
input_dataset, compared_dataset = create_dataset(train_set, TRAINING_TIME_INTERVAL_NUM)
train_input = input_dataset.reshape(-1, 1, TRAINING_TIME_INTERVAL_NUM)
train_compared = compared_dataset.reshape(-1, 1, 1)
train_input = torch.from_numpy(train_input)
train_compared = torch.from_numpy(train_compared)

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr = LR)
loss_func = torch.nn.MSELoss()
h_state = None

for step in range(1000):
    x = Variable(train_input)
    y = Variable(train_compared)

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data

    loss = loss_func(prediction, y)  # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()
    if (step + 1) % 100 == 0:
        print('Epoch: {}, Loss:{:.5f}'.format(step + 1, loss.item()))

rnn = rnn.eval()
input_dataset = input_dataset.reshape(-1, 1, TRAINING_TIME_INTERVAL_NUM)
input_dataset = torch.from_numpy(input_dataset)
prediction, h_state = rnn(input_dataset, h_state)
prediction = prediction.data.numpy()

plt.plot(prediction.flatten(), 'b', label = 'RNN Predicted Passenger Flow')
plt.plot(train_set.flatten(), 'r', label = 'Real Passenger Flow')
plt.xlabel('Time Interval : 15 minutes / Duration: April 1st to April 30th')
plt.ylabel('Passenger Number')
plt.title('Passenger Flow for UQ Lake Station [1882] during April 2019')
plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict = {'size':10, 'color':'red'})
plt.legend(loc='best')
plt.savefig('result.png', format='png', dpi=200)
plt.show()

