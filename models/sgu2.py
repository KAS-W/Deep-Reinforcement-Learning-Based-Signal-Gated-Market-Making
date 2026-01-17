import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SGU2Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SGU2Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, time_steps, input_size)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class SGU2:
    def __init__(self, input_size=1, hidden_size=10, device='cpu'):
        self.device = torch.device(device)
        self.model = SGU2Model(input_size, hidden_size).to(self.device)
        self.best_state = None 

    def train(self, X_train, y_train, X_val, y_val, batch_size=128, epochs=100, lr=0.001):
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience = 5
        trigger_times = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_state = self.model.state_dict() 
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        return history

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_t).cpu().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)