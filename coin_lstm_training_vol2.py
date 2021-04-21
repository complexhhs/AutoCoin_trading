#%%
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

if not os.path.isdir("./coin_nn_model"):
    os.makedirs("./coin_nn_model")

f_names = os.listdir("./coin_data")


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        # if we should predict the future
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


criterion = nn.MSELoss()

# parameters
epochs = 3

norm_file = open(os.path.join("./coin_nn_model", "coin_norm2.txt"), "w")
norm_file.write("coin_name    max_value    min_value")

dataset = np.empty((0, 49610), float)
for f in f_names:
    coin_name = os.path.splitext(f)[0].split("_")[0]

    if coin_name == "SNT":
        continue

    df = pd.read_excel(os.path.join("./coin_data", f))
    print(f"Training Object coin name: {coin_name}")

    data = df["target"].to_numpy().reshape(1, -1)
    max_data = data.max()
    min_data = data.min()
    data = (data - min_data) / (max_data - min_data)
    norm_file.write(
        str(coin_name) + "    " + str(max_data) + "    " + str(min_data)
    )
    data = data[:, :49610]
    dataset = np.append(dataset, data, axis=0)
    # print(data.shape)
norm_file.close()

input = torch.from_numpy(dataset[: int(len(dataset) * 0.8), :-1])
target = torch.from_numpy(dataset[: int(len(dataset) * 0.8), 1:])
test_input = torch.from_numpy(dataset[int(len(dataset) * 0.8) :, :-1])
test_target = torch.from_numpy(dataset[int(len(dataset) * 0.8) :, 1:])

# Training
seq = Sequence().double()
optimizer = optim.LBFGS(seq.parameters(), lr=0.8, max_iter=15)
# optimizer = optim.Adam(seq.parameters(), lr=1e-01)
model_name = os.path.join("./coin_nn_model", "model.pth")
for e in range(epochs):
    print(f"Epoch: {e+1}")
    # Training
    seq.train()

    def closure():
        out = seq(input)
        loss = criterion(out, target)
        print(f"loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(closure)
    # begin predict, no need to track gradient here
    seq.eval()
    with torch.no_grad():
        future = 1000  # data.shape[1]
        pred = seq(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print(f"Test loss: {loss.item()}")
    torch.save(seq, model_name)
print("")
