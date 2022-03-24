from sklearn.ensemble import RandomForestRegressor
from joblib import dump

class RandomForest:
    def __init__(self, max_depth, random_state, n_estimators):
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_estimators = n_estimators
        
    def fit(self, X, y):
        self.model = RandomForestRegressor(max_depth=self.max_depth, random_state=self.random_state, n_estimators=self.n_estimators)
        self.model.fit(X, y)

    def save_model(self, path):
        self.dump(self.model, path+'.joblib')