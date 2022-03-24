from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

class CNN_1D:
    def __init__(self, len):
        self.len = len
    
    def create_model(self):
        model = Sequential([
            # input layer
            Input(shape=(self.len, 1)),
            # 1D convolutional layer
            Conv1D(filters=32, kernel_size=4, strides=2, activation='relu'),
            # max pooling layer
            MaxPooling1D(pool_size=2, strides=2),
            # 1D convolutional layer
            Conv1D(filters=32, kernel_size=4, strides=2, activation='relu'),
            # max pooling layer
            MaxPooling1D(pool_size=2, strides=2),
            # three fully connected layers of sizes 128, 64 and 32
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='sigmoid')
        ])
        return model
        
    def fit(self, X, y, callbacks):
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mse'])
        callbacks = [ModelCheckpoint(filepath='./checkpoints/checkpoint', monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)]
        self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=2, callbacks=callbacks, shuffle=True)

    def save_model(self, path):
        self.model.save(path+'.h5')