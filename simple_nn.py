import numpy as np

# -----------------------------
# 1. Prepare dataset (XOR)
# -----------------------------
# Four inputs: (0,0), (0,1), (1,0), (1,1)
# Corresponding outputs (XOR): 0, 1, 1, 0
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

# -----------------------------
# 2. Hyperparameters
# -----------------------------
input_size = 2      # two input features
hidden_size = 2     # number of neurons in hidden layer
output_size = 1     # single output
learning_rate = 0.1
epochs = 5000

# -----------------------------
# 3. Initialize weights
# -----------------------------
# W1: (2x2), b1: (1x2)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

# W2: (2x1), b2: (1x1)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# -----------------------------
# 4. Activation function
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    # derivative of sigmoid with respect to z
    return sigmoid(z) * (1 - sigmoid(z))

# -----------------------------
# 5. Training loop
# -----------------------------
for epoch in range(epochs):
    # ---- Forward pass ----
    # Layer 1
    z1 = X.dot(W1) + b1        # shape: (4, 2)
    a1 = sigmoid(z1)           # shape: (4, 2)
    
    # Layer 2
    z2 = a1.dot(W2) + b2       # shape: (4, 1)
    a2 = sigmoid(z2)           # shape: (4, 1)  (this is our prediction)
    
    # Compute mean squared error (MSE) loss
    loss = np.mean((y - a2) ** 2)
    
    # ---- Backward pass ----
    # Gradient of loss w.r.t a2
    d_loss_a2 = 2 * (a2 - y) / y.shape[0]  # shape: (4, 1)
    
    # Gradient w.r.t z2
    d_loss_z2 = d_loss_a2 * sigmoid_deriv(z2)
    
    # Gradient w.r.t W2, b2
    d_loss_W2 = a1.T.dot(d_loss_z2)  # shape: (2, 1)
    d_loss_b2 = np.sum(d_loss_z2, axis=0, keepdims=True)  # shape: (1, 1)
    
    # Gradient w.r.t a1
    d_loss_a1 = d_loss_z2.dot(W2.T)  # shape: (4, 2)
    
    # Gradient w.r.t z1
    d_loss_z1 = d_loss_a1 * sigmoid_deriv(z1)
    
    # Gradient w.r.t W1, b1
    d_loss_W1 = X.T.dot(d_loss_z1)       # shape: (2, 2)
    d_loss_b1 = np.sum(d_loss_z1, axis=0, keepdims=True)  # shape: (1, 2)
    
    # ---- Update weights ----
    W2 -= learning_rate * d_loss_W2
    b2 -= learning_rate * d_loss_b2
    W1 -= learning_rate * d_loss_W1
    b1 -= learning_rate * d_loss_b1
    
    # Optional: print loss every 500 iterations
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# -----------------------------
# 6. Test the trained network
# -----------------------------
print("\nFinal predictions after training:")
final_output = sigmoid(sigmoid(X.dot(W1) + b1).dot(W2) + b2)
for i, (inp, out) in enumerate(zip(X, final_output)):
    print(f"Input: {inp}, Predicted: {out[0]:.4f}")
