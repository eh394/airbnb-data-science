import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
import datetime
import os
import json
from sklearn.model_selection import train_test_split


from tabular_data import load_airbnb, rating_columns, default_value_columns

np.random.seed(2)


df = pd.read_csv('clean_tabular_data.csv')
features, labels = load_airbnb(
    df, (rating_columns + default_value_columns), 'Price_Night')

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.7, test_size=0.3)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5)





class AirbnbNightlyPriceRegressionDataset(Dataset):
    
    def __init__(self, data):
        super().__init__()
        # data = pd.read_csv("clean_tabular_data.csv")
        # print(data)
        X_train, y_train = data
        # self.X = data[rating_columns + default_value_columns].astype(np.float32)
        # self.Y = data["Price_Night"].astype(np.float32)
        self.X = X_train.astype(np.float32)
        self.Y = y_train.astype(np.float32)
        

    def __getitem__(self, index):
        features = torch.tensor(self.X.iloc[index])
        label = self.Y.iloc[index]
        return (features, label)
    
    def __len__(self):
        return len(self.Y)
    


train_data = X_train, y_train
train_dataset = AirbnbNightlyPriceRegressionDataset(train_data)
validation_data = X_validation, y_validation
validation_dataset = AirbnbNightlyPriceRegressionDataset(validation_data)



# dataset = AirbnbNightlyPriceRegressionDataset()
# print(dataset[10])
# print(len(dataset))

# train_dataset, validation_dataset = random_split(dataset, [0.7, 0.3])
# validation_dataset, test_dataset = random_split(validation_dataset, [0.5, 0.5])



def get_nn_config(filename):
    with open(filename, "r") as params:
        params = yaml.safe_load(params)
        return params


params = get_nn_config("nn_config.yaml")


def train(model,  hyperparams, epochs=10):
    
    optimiser = getattr(torch.optim, hyperparams["optimiser"])
    optimiser = optimiser(model.parameters(), lr=hyperparams["learning_rate"])

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

        # validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True)
        # for batch in validation_loader:
        #     features, labels = batch
        #     prediction = model(features)
        #     loss = F.mse_loss(prediction, labels)
        #     print(loss)
    return model


class NN(torch.nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
         
        n = hyperparams["model_depth"] - 2

        layers = [torch.nn.Linear(10, hyperparams["hidden_layer_width"]), torch.nn.ReLU()]
        for i in range(n):
            layers.append(torch.nn.Linear(hyperparams["hidden_layer_width"], hyperparams["hidden_layer_width"]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hyperparams["hidden_layer_width"], 1))

        self.layers = torch.nn.Sequential(*layers)


    def forward(self, X):
        # returns prediction
        return self.layers(X)




train_data = X_train, y_train
train_dataset = AirbnbNightlyPriceRegressionDataset(train_data)
print(type(train_dataset))
# train_dataset, validation_dataset = random_split(dataset, [0.7, 0.3])
# validation_dataset, test_dataset = random_split(validation_dataset, [0.5, 0.5])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
model = NN(params)
train_start_time = datetime.datetime.now()
train(model, params)
train_end_time = datetime.datetime.now()
training_duration = train_end_time - train_start_time
print(training_duration)





print(type(model))

# torch.save(model.state_dict(), 'model.pt')
# state_dict = torch.load('model.pt')
# loaded_model = NN()
# loaded_model.load_state_dict(state_dict)

def save_model(trained_model, folder, opt_hyperparams=None, metrics=None):
    
    if str(type(trained_model)) == "<class '__main__.NN'>":
        now = datetime.datetime.now()
        folder = folder + f"{now}"
        os.makedirs(folder)
        filepath = folder + f"/model.pt"
        torch.save(model.state_dict(), filepath)

    else:
        filepath = folder + 'classification_model.joblib'
        joblib.dump(trained_model, filepath)

    filepath = folder + '/hyperparameters.json'
    json.dump(opt_hyperparams, open(filepath, 'w'))

    filepath = folder + '/metrics.json'
    json.dump(metrics, open(filepath, 'w'))


# save_model(model, "models/neural_networks/regression/")

# model.eval()
# validation_dataset = 
# X_validation, y_validation = validation_dataset
# print(validation_dataset)