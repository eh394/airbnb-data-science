import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


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
    

dataset = AirbnbNightlyPriceRegressionDataset()
print(dataset[10])
print(len(dataset))

train_dataset, validation_dataset = random_split(dataset, [0.7, 0.3])
validation_dataset, test_dataset = random_split(validation_dataset, [0.5, 0.5])



train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
example = next(iter(train_loader))
print(example)
features, labels = example
print('hello')
print(features.dtype)
print(labels.dtype)


class LinearRegression(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # initialise parameters
        self.linear_layer = torch.nn.Linear(10, 1)
        pass

    def forward(self, features):
        # use layers to process features
        return self.linear_layer(features)

model = LinearRegression()
print(model(features))



def train(model, epochs=10):
    
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    writer = SummaryWriter()

    batch_idx = 0
    
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1



train(model)




# for batch in train_loader:
#     print(batch)
#     features, labels = batch
#     print(features.shape)
#     print(labels.shape)
#     break