import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMNet(nn.Module):
    def __init__(self, input_size=23, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # self.fc = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn_input(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)
    
class SGU2:
    """LSTM for signal gate unit 2"""

    def __init__(self, input_size=23, hidden_size=64, num_layers=2, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMNet(input_size, hidden_size, num_layers).to(self.device)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train, y_train, X_val, y_val, batch_size=128, epochs=100, early_stopping_rounds=10):
        train_loader = self._prepare_loader(X_train, y_train, batch_size)
        val_loader = self._prepare_loader(X_val, y_val, batch_size)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        stop_count = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                # mse_loss = self.criterion(outputs.squeeze(), batch_y)
                # zero_mask = (batch_y == 0).float()
                # zero_penalty = torch.mean(zero_mask * torch.abs(outputs.squeeze()))
                # direction_penalty = torch.mean(torch.relu(-outputs.squeeze() * batch_y))
                # loss = mse_loss + 0 * zero_penalty + 0 * direction_penalty 
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
            
                # clip gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = self._evaluate_loss(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                stop_count = 0
                self.best_state = self.model.state_dict()
            else:
                stop_count += 1
                if stop_count >= early_stopping_rounds:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_state)
                    break
        
        return history

    def _prepare_loader(self, X, y, batch_size):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    
    def _evaluate_loss(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                outputs = self.model(batch_X)
                total_loss += self.criterion(outputs.squeeze(), batch_y).item()
        return total_loss / len(loader)
    
    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy().flatten()
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))