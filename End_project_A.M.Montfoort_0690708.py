#Import packages
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import ast

# Download Bitcoin data from Yahoo Finance
def download_data(crypto):
    if crypto == 'A':
        crypto_data = 'BTC-USD'
    elif crypto == 'B':
        crypto_data = 'ETH-USD'
    elif crypto == 'C':
        crypto_data = 'XRP-USD'
    else:
        print('None selected')

    data = yf.download(crypto_data, start='2017-11-09', end='2025-01-01', progress=False)
    return data

def train_val_test_split(data):
        data = data[['Close']]

        tvts = TimeSeriesSplit(n_splits = 3)

        for train_index, test_index in tvts.split(data):

            split_ratio = int(len(train_index)*0.8)         #80% train, 20% validation

            train_set = data.iloc[train_index[split_ratio:]]
            val_set = data.iloc[train_index[:split_ratio]]
            test_set = data.iloc[test_index]

        return train_set, val_set, test_set

def normalize_data(data):
    max_value = np.max(data)
    min_value = np.min(data)

    norm_data = (data - min_value) / (max_value - min_value)

    return norm_data, min_value, max_value

def x_y_data(data,input_layer):
    sequence_length = input_layer
    x_data, y_data = [], []

    for i in range(sequence_length, len(data)):
        x_data.append(data[i-sequence_length:i].values)
        y_data.append(data.values[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1])
    return x_data, y_data

def delta_act(X, activation):
    if activation == 'A':              #Hyperbolic tangent
         return np.tanh(X)
    if activation == 'B':
        return 1 / (1 + np.exp(-X))    #Sigmoid
    if activation == 'C':
        return np.maximum(0, X)        #ReLu
    
def delta_deriv(X, activation):
    if activation == 'A':              #Derivative hyperbolic tangent
        return 1 - np.tanh(X)**2
    if activation == 'B':
        return X*(1-X)                 #Derivative sigmoid
    if activation == 'C':
        return np.where(X>0,1,0)       #Derivative ReLu
    
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Initialize weights and biases for each layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.random.randn(layer_sizes[i+1]))

    def forward(self, X):
        self.layer_outputs = [X]  # Store inputs and outputs of each layer
        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[i], self.weights[i]) + self.biases[i]
            a = delta_act(z, activation)
            self.layer_outputs.append(a)

        return self.layer_outputs[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]  # Number of samples
        y_pred = self.layer_outputs[-1]
 
        # Calculate output layer error
        output_error = y_pred - y
        output_delta = output_error * delta_deriv(y_pred, activation)

        # Backpropagate the error
        deltas = [output_delta]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * delta_deriv(self.layer_outputs[i], activation)
            deltas.insert(0, delta)

        # Update weights and biases using gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.layer_outputs[i].T, deltas[i]) / m
            self.biases[i] -= learning_rate * np.mean(deltas[i], axis=0)

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate, patience=5):
        best_val_loss = float('inf')
        early_stop = 0

        for epoch in range(epochs):
            self.forward(X_train)  # Forward pass
            self.backward(X_train, y_train, learning_rate)  # Backward pass

            # Compute the loss on training data
            train_loss = np.mean(np.square(y_train - self.layer_outputs[-1]))  # Mean squared error

            # Evaluate on validation set
            val_predictions = self.forward(X_val)
            val_loss = np.mean(np.square(y_val - val_predictions))  # Mean squared error

            # Print current loss
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

            #Early stopping (only used for validation)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop = 0  
            else:
                early_stop += 1

            if early_stop >= patience:
                print(f"Early stopping at epoch {epoch} because no improvement in validation loss")
                break

    def test(self, X_test, min_test, max_test):
        prediction = self.forward(X_test)

        # Inverse normalization
        predicted_price = prediction * (max_test - min_test) + min_test
        return predicted_price[len(predicted_price) - 1]
        #return predicted_price[0]

    def pred_error(self, X_test):
        return self.forward(X_test)

    def norm_back(self, y_test, min_test, max_test):
        norm_back_result = y_test * (max_test - min_test) + min_test

        return norm_back_result[len(norm_back_result) - 1]
        #return norm_back_result[0]

# Inputs needed
crypto = input("Which cryptocurrency do you want to predict?: \n A) Bitcoin B) Ethereum C) XRP [A/B/C]? : ")
activation = input("Which activation function do you want to use?: \n A) Hyperbolic tangent B) Sigmoid C) ReLu [A/B/C]? : ")
epochs = int(input("How many epochs?: "))
learning_rate = float(input("Which learning rate?: "))
input_size = int(input("How many neurons in the input layer?: "))
hidden_sizes = ast.literal_eval(input("How many hidden layers and how many neurons in each hidden layers? \n (format: [30,20] means 30 neurons in the first hidden layer, 20 neurons in the second): "))

# Load the data, split the data into train, validation, and test sets
data = download_data(crypto)
train_set, val_set, test_set = train_val_test_split(data)
norm_train_set, min_train, max_train = normalize_data(train_set)
norm_val_set, min_val, max_val = normalize_data(val_set)
norm_test_set, min_test, max_test = normalize_data(test_set)

# Normalize the data
X_train, y_train = x_y_data(norm_train_set, input_size)
X_val, y_val = x_y_data(norm_val_set, input_size)
X_test, y_test = x_y_data(norm_test_set, input_size)

# Test: predictions
nn = NeuralNetwork(input_size, hidden_sizes, output_size=1)
nn.train(X_train, y_train, X_val, y_val, epochs, learning_rate)
#test_predictions = nn.test(X_test, max_test, min_test)
test_predictions = nn.test(X_test, min_test, max_test)
print("Test Predictions:")
print(test_predictions)
y_error_res = nn.pred_error(X_test)
test_loss = mse_loss(y_test, y_error_res)
print("Test error/loss:")
print(test_loss)
print('y_test:')
print(nn.norm_back(y_test, min_test, max_test))

# Plotting
actual_data = pd.DataFrame(data['Close'])
actual_data.columns = ['Price']
predict_data = pd.DataFrame({'Price': [test_predictions]})
all_data = pd.concat([actual_data, predict_data], ignore_index=True)
date_range = pd.date_range(start='2017-11-09', periods=len(all_data), freq='D')
all_data.index = date_range
plt.figure(figsize=(10, 6))
plt.plot(all_data[1:len(all_data) - 2], label='Actual Price')
plt.plot(date_range[len(all_data) - 1], all_data.iloc[len(all_data) - 1], 'ro', label='Predicted price')
plt.legend()
plt.title('Price Prediction')
plt.show()

# Trading strategy
if crypto == 'A':
    crypto_data = 'BTC-USD'
elif crypto == 'B':
    crypto_data = 'ETH-USD'
elif crypto == 'C':
    crypto_data = 'XRP-USD'
else:
    print('None selected')
print('Predicted price higher than half year average?: ')
half_year_avg = np.mean(yf.download(crypto_data, start='2024-06-01', end='2025-01-01', progress=False)['Close'])
print(test_predictions > half_year_avg)
print('Predicted price higher than last 3 month average?: ')
three_month_avg = np.mean(yf.download(crypto_data, start='2024-10-01', end='2025-01-01', progress=False)['Close'])
print(test_predictions > three_month_avg)
print('Predicted price higher than year average?: ')
year_avg = np.mean(yf.download(crypto_data, start='2024-01-01', end='2025-01-01', progress=False)['Close'])
print(test_predictions > year_avg)