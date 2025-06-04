import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")

#TF–IDF features
vectorizer     = TfidfVectorizer()
X_train_tfidf  = vectorizer.fit_transform(train_df["query"].tolist())
X_val_tfidf    = vectorizer.transform(val_df["query"].tolist())
X_test_tfidf   = vectorizer.transform(test_df["query"].tolist())

X_train = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
y_train = torch.tensor(train_df["carb"].values,      dtype=torch.float32)

X_val   = torch.tensor(X_val_tfidf.toarray(),   dtype=torch.float32)
y_val   = torch.tensor(val_df["carb"].values,  dtype=torch.float32)

X_test  = torch.tensor(X_test_tfidf.toarray(),  dtype=torch.float32)

# Dataset & DataLoader definitions
class NutriDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Define a single‐layer linear regressor
class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [batch_size, input_dim] → outputs [batch_size]
        return self.linear(x).squeeze(1)

# Hyperparameter configurations to try
configurations = [
    # Varying weight_decay
    {'optimizer': 'Adam', 'lr': 1e-4, 'batch_size': 16,  'weight_decay': 1e-5},
    {'optimizer': 'Adam', 'lr': 1e-4, 'batch_size': 16,  'weight_decay': 1e-4},
    {'optimizer': 'Adam', 'lr': 1e-4, 'batch_size': 16,  'weight_decay': 1e-6},
    {'optimizer': 'Adam', 'lr': 1e-4, 'batch_size': 16,  'weight_decay': 1e-7},

    # Varying batch_size
    {'optimizer': 'Adam', 'lr': 1e-4, 'batch_size': 32,  'weight_decay': 1e-5},
    {'optimizer': 'Adam', 'lr': 1e-4, 'batch_size': 64,  'weight_decay': 1e-5},
    {'optimizer': 'Adam', 'lr': 1e-4, 'batch_size': 128, 'weight_decay': 1e-5},

    # Varying lr
    {'optimizer': 'Adam', 'lr': 1e-3, 'batch_size': 16,  'weight_decay': 1e-5},
    {'optimizer': 'Adam', 'lr': 1e-5, 'batch_size': 16,  'weight_decay': 1e-5},
    {'optimizer': 'Adam', 'lr': 1e-6, 'batch_size': 16,  'weight_decay': 1e-5},
]

results = []
global_best_rmse   = float('inf')
global_best_config = None

input_dim = X_train.shape[1]  # number of TF–IDF features

# Training and validation utility functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)             # [batch_size]
        loss  = criterion(preds, yb)  # MSE loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_rmse(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds = model(Xb).cpu().numpy()  # [batch_size]
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())
    return np.sqrt(mean_squared_error(all_targets, all_preds))

#Loop over each configuration
for i, config in enumerate(configurations):
    print(f"\n{'='*60}")
    print(f"Configuration {i+1}: {config}")
    print(f"{'='*60}")

    # Create fresh model + criterion + optimizer
    model     = LinearRegressor(input_dim).to(device)
    criterion = nn.MSELoss()

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:
        raise NotImplementedError("Only Adam is supported in this loop.")

    # Create new DataLoaders with this batch_size
    train_loader = DataLoader(
        NutriDataset(X_train, y_train),
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        NutriDataset(X_val, y_val),
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Train with early stopping (patience = 10)
    best_val_rmse = float('inf')
    epochs_without_improvement = 0
    max_epochs = 50

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_rmse   = evaluate_rmse(model, val_loader, device)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"  Epoch {epoch+1:>2} | Train Loss = {train_loss:.4f} | Val RMSE = {val_rmse:.4f}")

        if epochs_without_improvement >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    #Record config's result
    results.append({
        'optimizer':      config['optimizer'],
        'lr':             config['lr'],
        'batch_size':     config['batch_size'],
        'weight_decay':   config['weight_decay'],
        'best_val_rmse':  best_val_rmse,
        'epochs_trained': epoch + 1
    })

    print(f"→ Config {i+1} best Val RMSE = {best_val_rmse:.4f}")

    # Check if this is the best overall; if so, save model + config
    if best_val_rmse < global_best_rmse:
        global_best_rmse   = best_val_rmse
        global_best_config = config.copy()
        torch.save(model.state_dict(), "best_linear_regressor.pth")

# After all configs, identify and save the best configuration
best_config = min(results, key=lambda x: x['best_val_rmse'])
print("\n" + "#"*60)
print("Best configuration overall:")
print(f"  optimizer:    {best_config['optimizer']}")
print(f"  lr:           {best_config['lr']}")
print(f"  batch_size:   {best_config['batch_size']}")
print(f"  weight_decay: {best_config['weight_decay']}")
print(f"  Val RMSE:     {best_config['best_val_rmse']:.4f}")
print(f"  Epochs used:  {best_config['epochs_trained']}")
print("#"*60)

# Save best_config to JSON for record
with open("best_config.json", "w") as f:
    json.dump(global_best_config, f, indent=4)
print("Saved optimizer configuration → best_config.json")
print("Saved best model state_dict → best_linear_regressor.pth")

# Load best model & predict on test set
best_model = LinearRegressor(input_dim).to(device)
best_model.load_state_dict(torch.load("best_linear_regressor.pth"))
best_model.eval()

# Use the same batch_size as the best configuration (or full‐batch if desired)
test_batch_size = best_config['batch_size']
test_loader = DataLoader(NutriDataset(X_test), batch_size=test_batch_size, shuffle=False)

all_test_preds = []
with torch.no_grad():
    for Xb in test_loader:
        Xb = Xb.to(device)
        preds = best_model(Xb).cpu().numpy()
        all_test_preds.extend(preds)

# 10. Save predictions to CSV
test_df["carb_pred"] = all_test_preds
output_csv = "test_with_best_predictions.csv"
test_df.to_csv(output_csv, index=False)
print(f"Saved test‐set predictions → {output_csv}")
