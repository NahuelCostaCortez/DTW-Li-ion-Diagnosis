# DTW-Li-ion-Diagnosis

This framework aims to enable the application of Deep Learning algorithms to battery degradation diagnosis by presenting a new way of representing cell degradation data. Particularly, the representation consists of an image highlighting the differences between the EVS curves of a pristine and aged battery with a DTW approach. The method is explored using a synthetic dataset to train a Convolutional Neural Network (CNN) that predicts the battery health state based on its degradation mechanisms.

The paper associated to this framework is published in Journal of Power Sources and is publicly available at https://www.sciencedirect.com/science/article/pii/S2352152X22015493.

You can also find a Gradio demo of the model in https://huggingface.co/spaces/NahuelCosta/DTW-CNN

# Files in this Repository
- \models: folder containing the implementation of different diagnosis models.
- \saved: folder containing the trained models.
- \mat: folder containing the data for the test sets.
- dtwRepresentation.ipynb: notebook containing the implementation of the DTW representation.
- trainDTW.py: training script for the CNN with the DTW images.
- train.py: script to train the rest of the models.
- utils.py: some helper functions.
- sweep.yaml: sweep configuration for wandb.
- requirements.txt: requirements for the project.

The training data used in this study is available for download.
- LFP: http://dx.doi.org/10.17632/bs2j56pn7y.
- NCA: http://dx.doi.org/10.17632/2h8cpszy26.1.
- NMC: http://dx.doi.org/10.17632/pb5xpv8z5r.1.

You can convert it to a python format with the save_data function in utils.py.

To execute the python code, it is recommended setting up a new python environment with packages matching the requirements.txt file.
It can be easily done with anaconda: conda create --name --file requirements.txt.
Another alternative is to run exactly the same environment under which this project was made. A Dockerfile is provided, which contains the set of instructions for creating a container with all the necessary packages and dependencies. The fastest way to set it up is to clone the reposity, open Visual Studio Code, and from the command palette select "Remote-containers: Open folder in Container".
