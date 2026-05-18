import numpy as np

# Dataset XOR
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(42)

# Bobot
W1 = np.random.rand(2,4)
W2 = np.random.rand(4,1)

lr = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

# Training
for epoch in range(10000):

    # Forward
    hidden = sigmoid(np.dot(X, W1))
    output = sigmoid(np.dot(hidden, W2))

    # Error
    error = y - output

    # Backprop
    d_output = error * sigmoid_derivative(output)

    hidden_error = d_output.dot(W2.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden)

    # Update
    W2 += hidden.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

print("Output akhir:")
print(np.round(output,3))