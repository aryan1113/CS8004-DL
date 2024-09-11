# -*- coding: utf-8 -*-
"""
Use mean square error for regression, with L2 regularization
This ensures, weights do not die down to zero as is the case with L1 regularization
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Helper Functions
# get mean square error between two numpy arrays
import numpy
def mse(y_true, y_pred):
    return numpy.mean(numpy.power(y_true-y_pred, 2))



# Load the California housing dataset
california = fetch_california_housing()

# Split the data into features and target
X = california.data
y = california.target

california_df = pd.DataFrame(california.data,
                             columns=california.feature_names) # Creates a data frame

california_df['MedHouseValue'] = pd.Series(california.target)

print(california.feature_names)
print(X.shape, y.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
#Transform each feature by scaling it based on the minimum and maximum values of the training data.

# Scale the training and test data
# Fun fact : We could scale the data before spillting
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

print(X_train.shape, y_train.shape)

# Define a linear regression model with 2 layers
class LinearRegression(nn.Module):
    def __init__(self, input):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(input, 4)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
      x = self.linear1(x)
      x = self.dropout(x)
      x = self.linear2(x)
      return x

# Well CPU is all I can give
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# Initialize the model
input = X_train.shape[1]
model = LinearRegression(input).to(device)
print(model)


# But why even use RMSprop, it does not perform well for sparse data as we do not get many updates in this direction
# Define loss function and optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
# change if RAM utilization breaches threshold
batch_size = 32
loss_fn = nn.MSELoss()

# Assuming model, optimizer, loss function (loss_fn), device, and data are already defined
epochs = 30  # Set the number of epochs
loss_history = []
val_loss_history = []
accuracy_history = []

for epoch in range(1, epochs + 1):
    # Set the model to training mode
    model.train()

    # Move data to device (GPU or CPU)
    X_train, y_train = X_train.to(device), y_train.to(device)

    # Forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # Zero gradients, backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Track loss
    loss_history.append(loss.item())

    # Calculate accuracy
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        X_test, y_test = X_test.to(device), y_test.to(device)
        predictions = model(X_test)
        test_loss = loss_fn(predictions, y_test)
        # print(f'Test Loss: {test_loss.item():.4f}')
        val_loss_history.append(test_loss.item())

    # Print loss and accuracy at every 100th epoch
    if (epoch) % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {test_loss:.2f}%')

# Plot the loss graph
plt.figure(figsize=(12, 5))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='b')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# Plot Training Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), val_loss_history, marker='o', linestyle='-', color='g')
plt.title('Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.show()