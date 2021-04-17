'''
sine wave lstm tutorial
Version 1.
Refer https://github.com/osm3000/Sequence-Generation-Pytorch/blob/master/models.py
Version 2.
Refer https://github.com/pytorch/examples/tree/master/time_sequence_prediction
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

def sine_train_ver1():
    '''
    parameters
    '''
    batch_size = 64
    seq_len = 100
    input_size=1
    hidden_size=51
    target_size=1
    nb_samples=1000
    nb_epochs_mainTraining = 2000
    nb_epochs_fineTuning=200
    
    x = np.linspace(0,60,600)
    y = np.sin(x)
    x_train = x[:int(len(x)*0.8)]
    y_train = y[:int(len(y)*0.8)]
    x_valid = x[int(len(x)*0.8):]
    y_valid = y[int(len(y)*0.8):]
    
    class dataset(torch.utils.data.Dataset):
        def __init__(self,x,y):
            self.x = x.reshape(-1,1)
            self.y = y.reshape(-1,1)
        
        def __getitem__(self,idx):
            return torch.Tensor(self.x[idx,:]),torch.Tensor(self.y[idx,:])
    
        def __len__(self):
            return len(self.x)
    
    train_dataset = dataset(x_train,y_train)
    valid_dataset = dataset(x_valid,y_valid)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=seq_len,shuffle=False,drop_last=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=seq_len,shuffle=False,drop_last=False)
        
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
    
            c_t = Variable(torch.zeros(input_data.size(0),self.hidden_size),requires_grad=False)
            c_t2 = Variable(torch.zeros(input_data.size(0),self.output_size),requires_grad=False)
    
            for i, input_t in enumerate(input_data.chunk(input_data.size(1),dim=1)):
                h_t, c_t = self.layer1(input_t,(h_t,c_t))
                h_t2,c_t2 = self.layer2(c_t,(h_t2,c_t2))
                outputs += [c_t2]
    
            for i in range(future):
                h_t,c_t = self.layer1(c_t2,(h_t,c_t))
                h_t2,c_t2 = self.layer2(c_t,(h_t2,c_t2))
                outputs += [c_t2]
    
            outputs = torch.stack(outputs,1).squeeze(2)
            return outputs
    
    
    rnn = lstm_rnn(input_size=input_size,hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(),lr=1e-02,weight_decay=1e-09)
    
    for epoch in range(nb_epochs_mainTraining):
        train_loss = 0
        rnn.train()
        for x,y in train_dataloader:
            pred = rnn(x)
            optimizer.zero_grad()
            loss = criterion(pred,y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
    
        rnn.eval()
        val_loss = 0
        with torch.no_grad():
            for x,y in valid_dataloader:
                pred = rnn(x)
                loss = criterion(pred,y)
                val_loss += loss.item()
        val_loss /= len(valid_dataloader)
    
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Train_loss: {train_loss}, Valid_loss: {val_loss}')
    
    torch.save(rnn,'sine_wave.pth') 
    

def sine_train_ver2():
    data = np.load('./Training_sin.npy')

    class Sequence(nn.Module):
        def __init__(self):
            super(Sequence,self).__init__()
            self.lstm1 = nn.LSTMCell(1,51)
            self.lstm2 = nn.LSTMCell(51,51)
            self.linear = nn.Linear(51,1)

        def forward(self,input,future=0):
            outputs = []
            h_t = torch.zeros(input.size(0),51,dtype=torch.double)
            c_t = torch.zeros(input.size(0),51,dtype=torch.double)
            h_t2 = torch.zeros(input.size(0),51,dtype=torch.double)
            c_t2 = torch.zeros(input.size(0),51,dtype=torch.double)

            for input_t in input.split(1,dim=1):
                h_t,c_t = self.lstm1(input_t,(h_t,c_t))
                h_t2,c_t2 = self.lstm2(h_t,(h_t2,c_t2))
                output = self.linear(h_t2)
                outputs += [output]

            # if we should predict the future
            for i in range(future): 
                h_t,c_t = self.lstm1(output,(h_t,c_t))
                h_t2,c_t2 = self.lstm2(h_t,(h_t2,c_t2))
                output = self.linear(h_t2)
                outputs += [output]
            outputs = torch.cat(outputs,dim=1)
            return outputs
    
    input = torch.from_numpy(data[3:,:-1])
    target = torch.from_numpy(data[3:,1:])
    test_input = torch.from_numpy(data[:3,:-1])
    test_target = torch.from_numpy(data[:3,1:])

    seq = Sequence().double()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(),lr=0.8)

    for e in range(15):
        print(f'Epoch: {e+1}')
        # Training
        optimizer.zero_grad()
        out = seq(input)
        loss = criterion(out,target)
        print(f'loss: {loss.item()}')
        loss.backward()
        optimizer.step()

        # begin predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input,future=future)
            loss = criterion(pred[:,:-future],test_target)
            print(f'Test loss: {loss.item()}')
            y = pred.detach().numpy()

        # plt.fiture
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequenc \n(Dashlines are predict values)', fontsize=30)
        plt.xlabel('x',fontsize=20)
        plt.ylabel('y',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.plot(np.arange(input.size(1)),y[0][:input.size(1)],'r',linewidth=2.0)
        plt.plot(np.arange(input.size(1),input.size(1)+future),y[0][input.size(1):],'r:',linewidth=2.0)
        
        plt.plot(np.arange(input.size(1)),y[1][:input.size(1)],'g',linewidth=2.0)
        plt.plot(np.arange(input.size(1),input.size(1)+future),y[1][input.size(1):],'g:',linewidth=2.0)

        plt.plot(np.arange(input.size(1)),y[2][:input.size(1)],'b',linewidth=2.0)
        plt.plot(np.arange(input.size(1),input.size(1)+future),y[2][input.size(1):],'b:',linewidth=2.0)

        plt.savefig('predict%d.pdf'%e)
        plt.close()

def generate_sinewave():
    np.random.seed(2)
    T = 20
    L = 1000
    N = 100
    x = np.empty((N,L),'int64')

    x[:] = np.array(range(L)+np.random.randint(-4*T,4*T,N).reshape(N,1))
    data = np.sin(x/T).astype('float64')
    np.save('Training_sin',data)


if __name__ == '__main__':
    #sine_train_ver1()
    sine_train_ver2()
    #generate_sinewave()