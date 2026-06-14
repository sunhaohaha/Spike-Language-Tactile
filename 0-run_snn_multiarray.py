import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from datetime import datetime
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train_orig = pd.read_excel('X_train.xlsx', header=None).values
Y_train_orig = pd.read_excel('Y_train.xlsx', header=None).values.squeeze()
X_test = pd.read_excel('X_test.xlsx', header=None).values
Y_test = pd.read_excel('Y_test.xlsx', header=None).values.squeeze()

X_train, X_val, Y_train, Y_val = train_test_split(X_train_orig, Y_train_orig,
                                                 test_size=0.25, stratify=Y_train_orig,
                                                 random_state=42)

ros = RandomOverSampler(random_state=42)
X_train, Y_train = ros.fit_resample(X_train, Y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)
class LIFNeuronLayer(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_neurons,
        tau=20,
        threshold=1.0,
        refractory_period=5,
        dropout_rate=0.2
    ):
        super(LIFNeuronLayer, self).__init__()

        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.tau = tau
        self.threshold = threshold
        self.refractory_period = refractory_period
        self.dropout = nn.Dropout(dropout_rate)

        self.weights = nn.Parameter(torch.randn(num_inputs, num_neurons) * 0.1)

    def forward(self, x):
        membrane_potential = torch.matmul(x, self.weights)

        hard_spikes = (membrane_potential >= self.threshold).float()

        surrogate_scale = 10.0
        soft_spikes = torch.sigmoid(
            surrogate_scale * (membrane_potential - self.threshold)
        )
        spikes = hard_spikes.detach() - soft_spikes.detach() + soft_spikes

        spikes = self.dropout(spikes)

        return spikes

class SNN(nn.Module):
    def __init__(self, num_inputs, num_neurons, num_hidden_neurons=1024):
        super(SNN, self).__init__()
        self.lif_layer1 = LIFNeuronLayer(num_inputs, num_hidden_neurons)
        self.lif_layer2 = LIFNeuronLayer(num_hidden_neurons, num_neurons)
        self.fc = nn.Linear(num_neurons, 17).to(device)
        self.bn1 = nn.BatchNorm1d(num_hidden_neurons).to(device)
        self.bn2 = nn.BatchNorm1d(num_neurons).to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.lif_layer1(x)
        x = self.bn1(x)
        x = self.lif_layer2(x)
        x = self.bn2(x)
        return self.fc(x)

model = SNN(num_inputs=X_train.shape[1], num_neurons=1024, num_hidden_neurons=2048).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

best_val_loss = float('inf')
patience = 20
wait = 0
best_model_state = None

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, Y_train)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, Y_val)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

if best_model_state is not None:
    model.load_state_dict(best_model_state)

def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(Y.cpu().numpy(), predicted.cpu().numpy())
        return outputs.cpu(), predicted.cpu(), accuracy

train_outputs, train_predicted, train_accuracy = evaluate(model, X_train, Y_train)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

test_outputs, test_predicted, test_accuracy = evaluate(model, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")