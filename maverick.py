"""
The goal of project maverick is to be able to predict the molecular weight of a molecule, given other numerically represented properties.
Said numerical data will include the likes of the number of Hydrogen bond donors, number of Hydrogen bond acceptors, polar area, chemical complexity score, etc.
This is my first project whereby the model will learn to derive one molecular property (molecular weight) from other molecular properties (as listed above).
If done correctly, the model should be capable of adapting to learn how to derive another molecular property (other than molecular weight).
The names of these molecules will not be used for training, meaning there will be no need for SMILES string, molecular fingerprints, or any other molecular representations.
Note that this code could have been written better:
    1. The training code could have been placed in a function to allow for better control.
    2. The data preprocessing portion of the code could have been optimized for computational efficiency better.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim

"""DATA PREPROCESSING"""

# Open the csv file with the dataset
data = pd.read_csv('maverick_data.csv')

# Define all data to be used in training and turn them into lists
molecular_weight = list(data.mw)
polar_area = list(data.polararea)
complexity = list(data.complexity)
heavycnt = list(data.heavycnt)
h_bond_donor = list(data.hbonddonor)
h_bond_acceptor = list(data.hbondacc)
rotbond = list(data.rotbonds)

assert len(polar_area) == len(complexity) == len(heavycnt) == len(h_bond_donor) == len(h_bond_acceptor) == len(rotbond) == len(molecular_weight)

# Define all ID points in dataset (Not to be used in training, but to identify the molecule)
molecular_formula = list(data.mf)
inchikey = list(data.inchikey)
iupac_name = list(data.iupacname)

assert len(molecular_formula) == len(inchikey) == len(iupac_name)

# Data metrics, stats, graphs, and plots
sns.scatterplot(molecular_weight, complexity).set_title("Molecular weights")
plt.show()

mean = np.mean(molecular_weight)
median = np.median(molecular_weight)
mode = stats.mode(molecular_weight, axis=None)
data_range = np.ptp(molecular_weight)
maximum = np.amax(molecular_weight)
minimum = np.amin(molecular_weight)

print(
      "\nMean = {0}".format(mean),
      "\nMedian = {0}".format(median),
      "\nMode = {0}".format(mode),
      "\nRange = {0}".format(data_range),
      "\nMaximum = {0}".format(maximum),
      "\nMinimum = {0}".format(minimum),
)

# Normalize all the data to be used in training
datasets = []
new_datasets = []

datasets.extend((polar_area, complexity, heavycnt, h_bond_donor, h_bond_acceptor, rotbond, molecular_weight))

def normalizer(datasets):
        for dataset in datasets:
                new_dataset = [(datapoint - float(min(dataset))) / float((max(dataset)) - float(min(dataset))) for datapoint in dataset]
                new_datasets.append(new_dataset)
        return new_datasets

normalizer(datasets)

# Assign a new variable for the Y data (molecular weight) and delete that dataset from the list of datasets)
Y_data = new_datasets[-1]
del new_datasets[-1]

Y_data = np.array(Y_data)
Y_data = torch.from_numpy(Y_data)
Y_data = Y_data.float()

# Turn list into a numpy array and then into tensor and define it with a variable
new_datasets = np.array(new_datasets)
X_data = torch.from_numpy(new_datasets)
X_data = X_data.float()

# Transpose data into proper format dimensions (6 input features of 12480, so 12480 lists of 6 features each)
X_data = torch.transpose(X_data, 0, 1)

print ("Length of total dataset: %s" %len(X_data))

# Manually split both X and Y datasets into testing and training datasets (4 datasets in total)
def data_splitter(dataset, train_percent):
    dataset_len = int(len(dataset))

    split_length = int(train_percent * dataset_len)
    return dataset[:split_length], dataset[split_length:]

X_data_train, X_data_test = data_splitter(X_data, .9)

print ("Length of X data train set is: %s" %len(X_data_train))
print ("Length of X data test set is: %s" %len(X_data_test))

Y_data_train, Y_data_test = data_splitter(Y_data, .9)

print ("Length of Y data train set is: %s" %len(Y_data_train))
print ("Length of Y data test set is: %s" %len(Y_data_test))

# Turn input (X_data) and labels (Y_data) into Tensors
for dataset in tqdm(X_data_train):
  dataset = dataset.view(-1, 6)
  
for dataset in tqdm(X_data_test):
  dataset = dataset.view(-1, 6)

Y_data_train = Y_data_train.view(-1, 1)
Y_data_test = Y_data_test.view(-1, 1)

# X_data_train = torch.stack(X_data_train, dim = 1)
# X_data_test = torch.stack(X_data_test, dim = 1)

# A quick check to ensure the datasets are ready to be processed
assert len(X_data_train) == len(Y_data_train)
assert len(Y_data_test) == len(Y_data_test)

"""DEEP NEURAL NETWORK MODEL"""

class Neural_Network(nn.Module):
        def __init__(self):
                super(Neural_Network, self).__init__()
                self.layer_1 = nn.Linear(6, 5)
                self.layer_2 = nn.Linear(5, 4)
                self.layer_3 = nn.Linear(4, 3)
                self.layer_4 = nn.Linear(3, 2)
                self.layer_5 = nn.Linear(2, 1)

                self.sigmoid = nn.Sigmoid()
                self.relu = nn.ReLU()

        def forward(self, X):
                out_1 = self.relu(self.layer_1(X))
                out_2 = self.sigmoid(self.layer_2(out_1))
                out_3 = self.sigmoid(self.layer_3(out_2))
                out_4 = self.sigmoid(self.layer_4(out_3))
                prediction = self.relu(self.layer_5(out_4))
                return prediction

# Creating a variable for our model
model = Neural_Network()

"""COMPILING, FITTING AND RUNNING"""

# Construct loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Construct training loop
epochs = 1000

running_loss = 0.0
for epoch in range(epochs):

        # Define Variables
        prediction = model(X_data_train)
        loss = criterion(prediction, Y_data_train)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if epoch % 10 == 9:
            print('Epochs: %5d | Loss: %.3f' % (epoch + 1, running_loss / 10))
            running_loss = 0.0
            
# Making predictions
model.eval()

evaluations = model(X_data_test)

for evaluation in evaluations:
  evaluation = int(evaluation)
print (evaluations)

def unnormalizer(datasets):
        for datapoint in datasets:
                new_datapoint = (datapoint + float(max(dataset))) * float(max(dataset)) + float(min(dataset))
        return new_datapoint

print (unnormalizer(evaluations))

# # Un-normalize the predicted molecular weights
# molecular_weight_pred = []

# def un_normalizer(outputs):
#         for output in outputs:
#                 molecular_weight = (output - float(min(outputs))) / float((max(outputs)) - float(min(outputs)))
#                 molecular_weight_pred.append(molecular_weight)
#         return molecular_weight_pred
      
# print (un_normalizer(evaluation))

"""
Personal takeaways from this project:
  1. One of the biggest problems I ran into was figuring out how best to control the output variables/labels (Y_data)
  leaving the labels unscalled caused the loss to skyrocket from decimal values to the tens of thousands
  as a result, I decided to leave them scaled but unscale them after running predictions on the test data
  2. I manually split the datasets to practice, but could have just as easily used PyTorch's built in dataloaders
  3. This is one of the first projects I learned to directly manipulate tensors to fit the model at hand using .view and .stack.
"""