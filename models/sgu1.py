import xgboost as xgb

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
            verbose=True
        )

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)