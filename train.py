'''
Training of the selected models. 
No hyperparameter tuning is performed as they were implemented as stated in their corresponding papers.
'''

import numpy as np
import utils

import models

# ------------- DATA LOADING --------------
chemistry = "LFP"
size = 128
# load data from npy file
x_train = np.load("./data/x_train"+str(size)+"_"+chemistry+".npy")
x_train = utils.normalise_data(x_train, np.min(x_train), np.max(x_train))
y_train = np.load("./data/y_train"+str(size)+"_"+chemistry+".npy")

# ------------- RANDOM FOREST -------------
model = models.RandomForest.RandomForest(max_depth=10, random_state=42, n_estimators=100)
model.fit(x_train, y_train)
model.save_model("./models/model-RF_"+chemistry)

# ----------------- CNN_1D -----------------
model = models.CNN_1D.CNN_1D(size)
model.fit(x_train, y_train)
model.load_weights('./checkpoints/checkpoint')
model.save('./models/model-CNN_'+chemistry+'.h5')
model.save_model("./models/model-CNN_1D_"+chemistry)

# ------------------- FFN -------------------
model = models.FFN.FFN(size)
model.fit(x_train, y_train)
model.save_model("./models/model-FFN_"+chemistry)