import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from models import CNN_DTW
import utils

if __name__ == "__main__":
	# If wandb is used run the following in the terminal: 
	# 	wandb login	
	# 	wandb agent --project project_name --entity entity_name --count num_experiments sweep_id
	# Default hyperparameters
	hyperparameter_defaults = dict(
		size=128,
		batch_size=32,
		learning_rate=0.0001,
    )

	# Pass your defaults to wandb.init
	wandb.init(config=hyperparameter_defaults)
	# Access all hyperparameter values through wandb.config
	config = wandb.config

	# ------------------------------------------------DATA-----------------------------------------------
	chemistry = "LFP"
	size = 128
	# load data from npy file
	x_train = np.load("data/x_train_"+chemistry+"_DTW.npy")
	x_train = utils.normalise_data(x_train, np.min(x_train), np.max(x_train))
	y_train = np.load("data/y_train_"+chemistry+".npy")
	# --------------------------------------------------------------------------------------------------

	# ---------------------------------------------TRAINING---------------------------------------------
	model = CNN_DTW.CNN_DTW(size)
	model.summary()
	callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
				 tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/checkpoint', monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True),
				 WandbCallback()]

	# fit the model
	model.fit(x_train, y_train, LR=config["learning_rate"], batch_size=config["batch_size"], callbacks=callbacks)
	# --------------------------------------------------------------------------------------------------

	# --------------------------------------------EVALUATION--------------------------------------------
	model.load_weights('./checkpoints/checkpoint')
	rmse_LLI = utils.rmspe(y_train[:,0], model.predict(x_train)[:,0])
	rmse_LAMPE = utils.rmspe(y_train[:,1], model.predict(x_train)[:,1])
	rmse_LAMNE = utils.rmspe(y_train[:,2], model.predict(x_train)[:,2])
	wandb.log({'rmspe LLI': rmse_LLI})
	wandb.log({'rmspe LAMPE': rmse_LAMPE})
	wandb.log({'rmspe LAMNE': rmse_LAMNE})
	# --------------------------------------------------------------------------------------------------