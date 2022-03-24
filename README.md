# DTW-Li-ion-Diagnosis

AI in battery research in still in its infancy. This framework aims to enable the application of Deep Learning algorithms by presenting a new way of representing battery data, especifically cell degradation data. Particularly, the representation consists of an image highlighting the differences between the
EVS curves of a pristine and aged battery. The method is explored using a synthetic dataset to train a
Convolutional Neural Network (CNN) that predicts the battery health state based on its degradation mechanisms.

# Files in this Repository
- \data: samples with which to train the model.
- \models: folder containing different models implementation.
- \saved: folder containing trained models.
- experimentalResults.ipynb: Jupyter notebook to reproduce the saved models results.
- trainDTW.py: training of DTW images with a CNN.
- train.py: rest of the models training.
- utils.py: some helper functions.
- sweep.yaml: sweep configuration for wandb.
- requirements.txt: requirements for the project.

To execute the python code, it is recommended setting up a new python environment with packages matching the requirements.txt file. 