'''
sine wave lstm tutorial
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.Variable as Variable

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

x = np.linspace(0,60,600)
y = np.sin(x)

class lstm_rnn(nn.Module):
    def __init__(self,input_size=1,hidden_size=20,output_size=1,activation='tanh'):
        super(lstm_rnn,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.activation = activation

        self.layer1 = nn.LSTMCell(input_size=self.input_size,hidden_size=self.hidden_size)
        self.layer2 = nn.LSTMCell(input_size=self.hidden_size,hidden_size=self.output_size)

    def forward(self,input_data,future=0):
        outputs = []
        h_t = Variable(torch.zeros(input_data.size(0),self.hidden_size),requires_grad=False)
        h_t2 = Variable(torch.zeros(input_data.size(0),self.output_size),requires_grad=False)

        c_t = Variable(torch.zeros(input_data.size(0),self.hidden_size),requires_grad=True)
        c_t2 = Variable(torch.zeros(input_data.size(0),self.output_size),)