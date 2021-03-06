'''
Training of the selected models. 
No hyperparameter tuning is performed as they were implemented as stated in their corresponding papers.
'''

import numpy as np
import utils

from models import RandomForest, CNN_1D, FFN

# ------------- DATA LOADING --------------
chemistry = "LFP"
size = 128
# load data from npy file
x_train = np.load("./data/x_train_"+chemistry+".npy")
x_train = utils.normalise_data(x_train, np.min(x_train), np.max(x_train))
y_train = np.load("./data/y_train_"+chemistry+".npy")

# ------------- RANDOM FOREST -------------
model = RandomForest.RandomForest(max_depth=10, random_state=42, n_estimators=100)
model.fit(x_train, y_train)
model.save_model("./models/model-RF_"+chemistry)

# ------------------- FFN -------------------
model = FFN.FFN(size)
model.fit(x_train, y_train)
model.save_model("./models/model-FFN_"+chemistry)

# ----------------- CNN_1D -----------------
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
model = CNN_1D.CNN_1D(size)
model.fit(x_train, y_train)
model.load_weights('./checkpoints/checkpoint')
model.save_model("./models/model-CNN_1D_"+chemistry)