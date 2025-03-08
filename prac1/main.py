import _pickle as cp
import matplotlib.pyplot as plt
import numpy as np

# Load data
X, y = cp.load(open('data/winequality-white.pickle', 'rb'))

# Split into training and test sets
N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train
X_train, y_train = X[:N_train], y[:N_train]  # Takes the first 80%
X_test, y_test = X[N_train:], y[N_train:]

# --- Handin 1: Plot bar chart ---
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts, color='skyblue', edgecolor='black')
plt.xlabel("Wine Quality (y)")
plt.ylabel("Frequency")
plt.title("Distribution of Wine Quality Scores in Training Set")
plt.xticks(range(3, 10))  # Wine quality scores range from 3 to 9
save_path = "/app/distribution.png"
plt.savefig(save_path)  # Save image to the mapped directory
print(f"Bar chart saved as {save_path}")

# --- Handin 2: Compute MSE for Trivial Predictor ---
mean_y_train = np.mean(y_train)

# Compute MSE for training set
mse_train = np.mean((y_train - mean_y_train) ** 2)

# Compute MSE for test set
mse_test = np.mean((y_test - mean_y_train) ** 2)

print(f"Trivial Predictor MSE (Train): {mse_train:.4f}")
print(f"Trivial Predictor MSE (Test): {mse_test:.4f}")

# --- Standardization ---
mean_X_train = np.mean(X_train, axis=0)  # Compute mean per feature
std_X_train = np.std(X_train, axis=0)    # Compute std deviation per feature

# Avoid division by zero in case of constant features
std_X_train[std_X_train == 0] = 1  # Set zero std dev to 1 in training set

X_train_standardized = (X_train - mean_X_train) / std_X_train
X_test_standardized = (X_test - mean_X_train) / std_X_train  # Use training mean & std

# --- Closed-form Least Squares Solution ---
# Add bias term (column of ones) for intercept
X_train_aug = np.c_[np.ones(N_train), X_train_standardized]
X_test_aug = np.c_[np.ones(len(X_test)), X_test_standardized]

# Compute beta (closed-form solution) vector of coefficients (including the intercept term b_0)
beta = np.linalg.inv(X_train_aug.T @ X_train_aug) @ X_train_aug.T @ y_train

# Predictions
y_train_pred = X_train_aug @ beta
y_test_pred = X_test_aug @ beta

# --- Compute Mean Squared Error ---
mse_train = np.mean((y_train - y_train_pred) ** 2)
mse_test = np.mean((y_test - y_test_pred) ** 2)

print(f"Linear Model MSE (Train): {mse_train:.4f}")
print(f"Linear Model MSE (Test): {mse_test:.4f}")

# Initialize lists to store the errors
train_errors = []
test_errors = []

# Loop over different training sizes
for n in range(20, 601, 20):  # Starting from 20, incrementing by 20, up to 600
    X_train_subset = X_train[:n]
    y_train_subset = y_train[:n]

    # Standardize the subset of the training data (using the same means and stds as before)
    X_train_subset_standardized = (X_train_subset - mean_X_train) / std_X_train
    X_train_subset_standardized[:, std_X_train == 0] = 0  # Set columns with zero std dev to zero

    # Augment the training data (adding a column of ones for the intercept term)
    X_train_aug_subset = np.c_[np.ones(n), X_train_subset_standardized]

    # Train the model using the closed-form solution
    beta_subset = np.linalg.inv(X_train_aug_subset.T @ X_train_aug_subset) @ X_train_aug_subset.T @ y_train_subset

    # Make predictions on the training subset
    y_train_pred_subset = X_train_aug_subset @ beta_subset

    # Calculate the training error (Mean Squared Error)
    train_mse = np.mean((y_train_subset - y_train_pred_subset) ** 2)
    train_errors.append(train_mse)

    # Standardize the test data using the same mean and std as the training data
    X_test_standardized = (X_test - mean_X_train) / std_X_train
    X_test_standardized[:, std_X_train == 0] = 0  # Set columns with zero std dev to zero

    # Augment the test data
    X_test_aug = np.c_[np.ones(len(X_test)), X_test_standardized]

    # Make predictions on the test data
    y_test_pred = X_test_aug @ beta_subset

    # Calculate the test error (Mean Squared Error)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    test_errors.append(test_mse)

# Plot the learning curves
plt.plot(range(20, 601, 20), train_errors, label='Training Error', color='blue')
plt.plot(range(20, 601, 20), test_errors, label='Test Error', color='red')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Learning Curves (Training and Test Error)')
plt.legend()
# Rotate x-axis labels for better readability
# plt.xticks(rotation=45)

# Optional: You can also set the ticks manually for better spacing
# For example, if you want fewer tick marks:
plt.xticks(np.arange(20, 601, 100))  # Set fewer ticks

save_path = "/app/learning-curves.png"
plt.savefig(save_path)  # Save image to the mapped directory
