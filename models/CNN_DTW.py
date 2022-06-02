from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

class CNN_DTW:
    def __init__(self, size):
        self.size = size
        self.model = Sequential([
            # Note the input shape is the desired size of the image sizexsize with 1 color channel
            Masking(mask_value=-99.0, input_shape=(self.size, self.size, 1)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dense(3, activation='sigmoid')
        ])
        
    def fit(self, X, y, LR, batch_size, callbacks):
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=LR), metrics=['mse'])
        self.model.fit(X, y, epochs=500, batch_size=batch_size, validation_split=0.2, verbose=2, callbacks=callbacks, shuffle=True)

    def save_model(self, path):
        self.model.save(path+'.h5')