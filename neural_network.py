import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml


from tabular_data import load_airbnb, rating_columns, default_value_columns

np.random.seed(2)


# df = pd.read_csv('clean_tabular_data.csv')
# features, labels = load_airbnb(
#     df, (rating_columns + default_value_columns), 'Price_Night')

class AirbnbNightlyPriceRegressionDataset(Dataset):
    
    def __init__(self):
        super().__init__()
        data = pd.read_csv("clean_tabular_data.csv")
        # print(data)
        self.X = data[rating_columns + default_value_columns].astype(np.float32)
        self.Y = data["Price_Night"].astype(np.float32)
        

    def __getitem__(self, index):
        features = torch.tensor(self.X.iloc[index])
        label = self.Y.iloc[index]
        return (features, label)
    
    def __len__(self):
        return len(self.Y)
    

# dataset = AirbnbNightlyPriceRegressionDataset()
# print(dataset[10])
# print(len(dataset))

# train_dataset, validation_dataset = random_split(dataset, [0.7, 0.3])
# validation_dataset, test_dataset = random_split(validation_dataset, [0.5, 0.5])



# train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
# example = next(iter(train_loader))
# print(example)
# features, labels = example
# print('hello')
# print(features.dtype)
# print(labels.dtype)


# class LinearRegression(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         # initialise parameters
#         self.linear_layer = torch.nn.Linear(10, 1)
#         pass

#     def forward(self, features):
#         # use layers to process features
#         return self.linear_layer(features)

# model = LinearRegression()
# print(model(features))


def get_nn_config(filename):
    with open(filename, "r") as params:
        params = yaml.safe_load(params)
        return params


params = get_nn_config("nn_config.yaml")


def train(model,  hyperparams, epochs=10):
    
    optimiser = getattr(torch.optim, hyperparams["optimiser"])
    optimiser = optimiser(model.parameters(), lr=hyperparams["learning_rate"])

    # optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    writer = SummaryWriter()

    batch_idx = 0
    
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            # print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1

        validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True)
        for batch in validation_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            print(loss)


class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )


    def forward(self, X):
        # returns prediction
        return self.layers(X)




dataset = AirbnbNightlyPriceRegressionDataset()
train_dataset, validation_dataset = random_split(dataset, [0.7, 0.3])
validation_dataset, test_dataset = random_split(validation_dataset, [0.5, 0.5])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
model = NN()
train(model, params)

