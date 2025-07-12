
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Load data (adjust paths as needed)
books = pd.read_csv("books.csv")
ratings = pd.read_csv("ratings.csv")

# Encode User and Book IDs
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()
ratings["user"] = user_encoder.fit_transform(ratings["User-ID"])
ratings["book"] = book_encoder.fit_transform(ratings["ISBN"])

n_users = ratings["user"].nunique()
n_books = ratings["book"].nunique()

# Dataset
X_users = torch.tensor(ratings["user"].values, dtype=torch.long)
X_books = torch.tensor(ratings["book"].values, dtype=torch.long)
y = torch.tensor(ratings["Book-Rating"].values, dtype=torch.float32)
dataset = TensorDataset(X_users, X_books, y)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)


# 1. Matrix Factorization Model
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_books, n_factors=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.book_emb = nn.Embedding(n_books, n_factors)

    def forward(self, user, book):
        u = self.user_emb(user)
        b = self.book_emb(book)
        return (u * b).sum(1)


# 2. Content-Based MLP Regressor (example uses Year-Of-Publication)
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


# 3. Neural Collaborative Filtering
class NeuralCF(nn.Module):
    def __init__(self, n_users, n_books, emb_dim=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.book_emb = nn.Embedding(n_books, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user, book):
        u = self.user_emb(user)
        b = self.book_emb(book)
        x = torch.cat([u, b], dim=1)
        return self.mlp(x).squeeze()


# Training Function (for Matrix Factorization or NeuralCF)
def train_model(model, dataloader, epochs=5, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for users, books, ratings in dataloader:
            preds = model(users, books)
            loss = loss_fn(preds, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")


# Example usage
# model = MatrixFactorization(n_users, n_books)
# model = NeuralCF(n_users, n_books)
# train_model(model, loader)
