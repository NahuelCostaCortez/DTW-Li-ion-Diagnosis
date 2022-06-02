from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class FFN:
    def __init__(self, len):
        self.len = len
        self.model = Sequential([
            Dense(64, input_dim=self.len, activation='relu'),
            Dense(32, activation='relu'),
            Dense(3, activation='sigmoid')
        ])
        
    def fit(self, X, y):
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    def save_model(self, path):
        self.model.save(path+'.h5')