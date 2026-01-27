import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SGU1:
    """XGBoost Model for Signal Gate Unit 1"""

    def __init__(self, model_path=None):

        self.params = {
            'max_depth': 4,
            'min_child_weight': 4,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'learning_rate': 0.01,
            'reg_alpha': 0.01,
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'early_stopping_rounds': 20  
        }
        self.model = xgb.XGBRegressor(**self.params)
        self.model_path = model_path

    def train(self, X_train, y_train, X_val, y_val):
        """Train the model and implement early stopping"""
        self.model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            verbose=False
        )

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)

class SGU2Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SGU2Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, time_steps, input_size)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)

class SGU2:
    """
    LSTM model for SGU2
    """

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

    def load(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # here must add weights_only=True for current torch version
        # or there are will be long warning logs
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"SGU2 model loaded from {path} and set to eval mode.")