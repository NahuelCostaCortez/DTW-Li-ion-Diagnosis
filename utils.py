import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator as pchip
from scipy.spatial import KDTree
from dtaidistance import dtw
import h5py
import scipy.io as sio
from IPython.display import display
import pandas as pd

UI_STEP = 0.0005
MIN_V_LFP = 3.20
MAX_V_LFP = 3.50
MIN_V_NCA = 3.20
MAX_V_NCA = 4.23
MIN_V_NMC = 3.44
MAX_V_NMC = 4.28
SIZE = 128
LFP_MIN = 0.0031566024955984595
LFP_MAX = 2.736867845392978
NCA_MIN = 0
NCA_MAX = 0.2682065162447511
NMC_MIN = 0
NMC_MAX = 0.1914037352896408

# --------------------------------------------------READ DATA--------------------------------------------------
def read_mat(file_name):
    '''
    Reads a .mat file and returns the data as a numpy array

    Parameters
	----------
	file_name: str, path to the .mat file
    '''

    return sio.loadmat(file_name)

def read_mat_hdf5(file_name, field_name):
	'''
	Opens mat file as a numpy array with hdf5
    Must retrieve all the indexes: advanced indexing in h5py is not nearly as general as with np.ndarray,
    an exception will be raised if the indexes are not continuous

    Parameters
	----------
	file_name: str, path to the .mat file
    field_name: str, name of the field inside the .mat file
	'''

	with h5py.File(file_name, 'r') as f:
		data = f[field_name][:]
	return data

def get_V_reference(file_name, field_name):
	'''
    Returns the voltage curve of the reference cell (cycle 0)

    Parameters
	----------
	file_name: str, path to the .mat file
    field_name: str, name of the field inside the .mat file

	Returns
    -------
	V: voltage curve
    '''

	with h5py.File(file_name, 'r') as f:
		V = f[field_name][0]
	return V
# -------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------INDEXES---------------------------------------------------
def get_indexes(info, i, last_index, degradation_mode, resolution):
    '''
    Filters indexes according to a given resolution in a given path

    Parameters
    ----------
    info: numpy array, contains the labels for each curve
    i: int, index of the first sample in the path
    last_index: int, index of the last sample in the path
    degradation_mode: int, 0 for LLI, 1 for LAMPE, 2 for LAMNE
    resolution: int, resolution of the data

    Returns
    -------
    indexes: numpy array, contains the resulting indexes
    '''

    # take only the path data
    path = info[i:last_index]
    # from the path data take the LLI/LAMPE/LAMNE values from 0 to 80 with a resolution of 'resolution'%
    path = KDTree(path[:,degradation_mode].reshape(path.shape[0], 1))
    res_values = np.arange(0, 85, resolution)
    res_values = res_values.reshape(res_values.shape[0], 1)
    # indexes contains the indexes of the LLI/LAMPE/LAMNE values closest to the requested resolution 
    _, indexes = path.query(res_values, k=1)
    # get their global indexes inside the info matrix
    indexes = indexes + i
    return indexes

def get_curves(info, resolution):
    '''
    Filters indexes according to a given resolution

    Parameters
    ----------
    info: numpy array, contains the labels for each curve
    resolution: int, resolution of the data

    Returns
    -------
    numpy array, contains the resulting indexes
    '''

    # retrieve indices of the paths according to the main degradation
    indexes_LLI = np.where(info[:,0] == 85)[0]
    indexes_LAMPE = np.where(info[:,1] == 85)[0]
    indexes_LAMNE = np.where(info[:,2] == 85)[0]
    # array to save the selected indexes
    selected_indexes = []
    len_paths = 137
    
    # loop over the paths
    for i in range(1, len(info), len_paths):
        # last index of the path must be 85
        last_index = i + len_paths - 1
        if last_index in indexes_LLI:
            selected_indexes.append(get_indexes(info, i, last_index, 0, resolution))

        if i+137-1 in indexes_LAMPE:
            selected_indexes.append(get_indexes(info, i, last_index, 1, resolution))

        if i+137-1 in indexes_LAMNE:
            selected_indexes.append(get_indexes(info, i, last_index, 2, resolution))
            
    return  np.sort(np.array(selected_indexes).flatten())
# -------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------IC DATA--------------------------------------------------
def IC(u, q, ui_step=0.0005, minV=3.2, maxV=3.5):
    '''
    Get the ICA data for a given voltage curve

    Parameters
    ----------
    u: numpy array, voltage curve
    q: numpy array, capacity curve
    ui_step: float, step of interpolation
    minV: float, minimum voltage of the IC curve
    maxV: float, maximum voltage of the IC curve

    Returns
    -------
    ui, dqi: numpy arrays, interpolated voltage and derivative of capacity
    '''

    # voltages values for which capacity is interpolated
    ui = np.arange(minV, maxV, ui_step) 
    qi = np.interp(ui, u, q)
    return ui[1:], np.diff(qi)

def reduce_size(ui, dqi, size):
    '''
    Reduces the length of the IC data to a given size

    Parameters
    ----------
    ui: numpy array, voltage curve
    dqi: numpy array, derivative of capacity (IC)
    size: int, size at which to reduce the IC data

    Returns
    -------
    numpy array, reduced IC
    '''

    curve = pchip(ui, dqi)
    ui_reduced = np.linspace(min(ui), max(ui), size)
    return curve(ui_reduced)

def get_max_IC(v, q):
    '''
    Returns the maximum IC value

    Parameters
    ----------
    v: numpy array, voltage curve
    q: numpy array, capacity curve
    Both must correspond to the reference IC

    Returns
    -------
    float, maximum IC value
    '''

    # max voltage value of new cell IC
    return max(IC(v, q)[1])

def normalise_data(data, min_val, max_val, low=0, high=1):
    '''
    Normalises the data to the range [low, high]

    Parameters
    ----------
    data: numpy array, data to normalise
    min: float, minimum value of data
    max: float, maximum value of data
    low: float, minimum value of the range
    high: float, maximum value of the range

    Returns
    -------
    normalised_data: float, normalised data
    '''
    normalised_data = (data - min_val)/(max_val - min_val)
    normalised_data = (high - low)*normalised_data + low
    return normalised_data

def get_IC_samples(info, V, Q, max_IC, ui_step, minV, maxV, size):
    '''
    Returns the IC samples for each curve

    Parameters
    ----------
    info: numpy array, contains the labels for each curve
    V: numpy array, voltage curve
    Q: array, capacity percentages from 0 to 100 from the simulated dataset
    max_ICA: float, maximum IC value
    ui_step: float, step of the interpolation
    minV: float, minimum voltage of the IC
    maxV: float, maximum voltage of the IC
    size: int, size at which to reduce the IC data

    Returns
    -------
    info_ICs, ICs: numpy arrays, contains the labels for each curve and the IC samples
    '''

    samples = []
    new_info = []
    for curve, curve_info in zip(V, info):
        ui, dqi = IC(curve, Q, ui_step, minV, maxV)
        new_sample = reduce_size(ui, dqi, size)
        # if this height is exceeded, it is not considered a realistic situation, then it is omitted.
        if max(new_sample) < max_IC*3:
            new_info.append(curve_info)
            samples.append(new_sample)
    return np.array(new_info), np.array(samples)
# -------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------TRAINING DATA--------------------------------------------
def get_data(info, V, V_reference, Q, resolution, ui_step, minV, maxV, size):
    '''
    Returns training data

    Parameters
    ----------
    info: numpy array, contains the labels for each curve
    V_reference: numpy array, voltage curve of the reference cell (cycle 0)
    Q: array, capacity percentages from 0 to 100 from the simulated dataset
    resolution: int, resolution of data
    ui_step: float, step of interpolation
    minV: float, minimum voltage of the IC
    maxV: float, maximum voltage of the IC
    size: int, size at which to reduce the IC data

    Returns
    -------
    x, y: numpy arrays, contain the IC samples and the labels for each curve
    '''

    # 1. Select curves according to a given resolution
    selected_indexes = get_curves(info, resolution)
    # add the index corresponding to the reference curve
    selected_indexes = np.insert(selected_indexes, 0, 0, axis=0)

    # 2. Retrieve data from the selected indexes
    info = info[selected_indexes]
    V = V[selected_indexes]

    # 3. Prone according to the stated requirements
    indexes_to_remove = []
    # curves with LAMNE < 0
    LAMNE_negative = np.where(info[:,2] < 0)[0]
    indexes_to_remove.append(LAMNE_negative)
    # curves with capacity loss > 40
    capacity_loss_high = np.where(info[:,3] > 40)[0]
    indexes_to_remove.append(capacity_loss_high)
    # finally get unique indexes
    indexes_to_remove = np.unique(np.concatenate(indexes_to_remove))
    info = np.delete(info, indexes_to_remove, axis=0)
    V = np.delete(V, indexes_to_remove, axis=0)

    # convert voltage curves to IC
    max_IC = get_max_IC(V_reference, Q)
    info_ICs, ICs = get_IC_samples(info, V, Q, max_IC, ui_step, minV, maxV, size)
    return ICs, info_ICs[:,0:3]/100

def plot_path(V_reference, Q, ICs, info, sample_number, size):
    '''
    Plots a given path

    Parameters
    ----------
    V_reference: numpy array, voltage curve of the reference cell (cycle 0)
    Q: array, capacity percentages from 0 to 100 from the simulated dataset
    info: numpy array, contains the labels for each curve
    sample_number: int, number of the sample to plot
    size: int, size at which to reduce the IC data
    '''

    plt.figure(figsize=(18, 6))
    plt.title("Degradation vs reference IC")
    plt.plot(ICs[sample_number], label="LLI: "+ str(info[sample_number][0]) + "    LAMPE: " + str(info[sample_number][1]) + "    LAMNE: " + str(info[sample_number][2]) + "    Capacity Loss: " + str(info[sample_number][3]))
    ui, dqi = IC(V_reference, Q, 0.0005, 3.2, 3.5)
    IC_reference = reduce_size(ui, dqi, size)
    plt.plot(IC_reference, label="Reference")
    plt.legend()
    plt.show()

def get_DTWImages(data, reference, size):
    '''
    Converts IC samples to DTW images

    Parameters
    ----------
    data: numpy array, contains the IC samples
    reference: numpy array, contains the IC of the reference cell
    size: int, size at which to reduce the IC data -> resolution of the resulting image

    Returns
    -------
    x: numpy array, DTW images
    '''

    images = []
    for IC_sample in data:
        d, paths = dtw.warping_paths(reference, IC_sample, window=int(size/2), psi=2)
        images.append(paths)
    x = np.array(images)
    # mask values that are not filled
    x = np.where(x == np.inf, -99, x)
    # negative values are replaced by 0
    x = np.where(x < 0, 0, x)
    # normalise values
    x = x/np.max(x)
    # reshape the array
    x = np.expand_dims(x, -1).astype("float32")
    return x

def get_minmaxV(material):
    '''
    Returns the range voltage in which to study the IC curves

    Parameters
    ----------
    material: numpy array, chemistry to study

    Returns
    -------
    min_v, max_v, path: numpy arrays, min and max voltage values and path where data is located
    '''

    min_v = -1
    max_v = -1
    path = ""
    if material == "LFP":
        path = './mat/LFP/diagnosis'
        min_v = MIN_V_LFP
        max_v = MAX_V_LFP
    elif material == "NCA":
        path = './mat/NCA/diagnosis'
        min_v = MIN_V_NCA
        max_v = MAX_V_NCA
    elif material == "NMC":
        path = './mat/NMC/diagnosis'
        min_v = MIN_V_NMC
        max_v = MAX_V_NMC
    else:
        print("ERROR: Chemistry not found")
        return -1
    if min_v == -1 or max_v == -1 or path == "":
        print("ERROR: Chemistry not found")
        return -1
    return min_v, max_v, path

def save_data(size, material):
    '''
    Save data to disk

    Parameters
    ----------
    size: int, size at which to reduce the IC data
    material: numpy array, chemistry to study
    '''

    min_v, max_v, path = get_minmaxV(material)

    Q = read_mat('./mat/Q.mat')
    Q = Q['Qnorm'].flatten()
    info = read_mat(path+'/pathinfo.mat')['pathinfo']
    V = read_mat_hdf5(path+'/V.mat', 'volt')
    # voltage curve of the cell when the degradation is 0
    V_reference = get_V_reference(path+'/V.mat', 'volt')

    x_train, y_train = get_data(info, V, V_reference, Q, 2, UI_STEP, min_v, max_v, size-1)
    np.save("data/x_train"+str(size)+"_"+material+".npy", x_train)
    np.save("data/y_train"+str(size)+"_"+material+".npy", y_train)

def save_DTW_data(size, material):
    '''
    Saves DTW data to disk

    Parameters
    ----------
    size: int, resolution of the resulting image
    material: string, cell chemistry
    '''

    # ICs
    x = np.load("data/x_train"+str(size)+"_"+material+".npy")
    # Not really necessary but it´s done for computational efficiency
    x = normalise_data(x, np.min(x), np.max(x))

    # The reference IC is the first sample
    IC_reference = x[0]
    
    # Generate the DTW images
    x_DTW = get_DTWImages(x, IC_reference, size)

    # Save data
    np.save("data/x_train"+str(size)+"_"+material+"_DTW.npy", x_DTW)


# ----------------------------------------------------INFERENCE------------------------------------------------
def rmspe(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    return np.sqrt(np.mean(np.square((y_true - y_pred)), axis=0))*100

def get_IC_references_test(material, size, x_train_pre, Q, path):
	'''
	Returns the IC curves for the test sets

	Parameters
	----------
	material: str, chemistry of the battery
    size: int, size at which to reduce the IC data
	x_train_pre: array, needed for normalising the ICs the same way as in the training set
    Q: array, capacity percentages from 0 to 100 from the simulated dataset
    path: path where data is located
	'''
	min_v, max_v, _ = get_minmaxV(material)
	V_references = read_mat(path+'/V_references.mat')['V_references']
	IC_references = []
	for cell in V_references:
		ui, dqi = IC(cell, Q, UI_STEP, min_v, max_v)
		new_sample = reduce_size(ui, dqi, size-1)
		# it is necessary to normalise because the x_tests are also normalized in get_data_eval
		IC_references.append(normalise_data(new_sample, np.min(x_train_pre), np.max(x_train_pre)))
	return IC_references

def get_pred(model, x_tests, y_test, reshape, DTW):
	'''
	Prints predictions for test sets

	Parameters
	----------
	model: h5py object, trained model
    x_test: list, test sets
	y_test: array, test set labels
    reshape: bool, if True, an extra dimension is added to the input
    DTW: bool, if True, the data is reshaped to the DTW model input shape
	'''
	average = []
	y_test = y_test.reshape(-1, 6, y_test.shape[1])
    
	for x_test_pre in x_tests:
		cycles = [10, 50, 100, 200, 400, 1000]
		x_test = x_test_pre.reshape(-1, 6, x_test_pre.shape[1])

		if reshape == True:
			x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

		if DTW == True:
			x_test = x_test.reshape(-1, 6, x_test_pre.shape[1], x_test_pre.shape[2], 1)

		predictions_LLI = np.zeros(len(cycles))
		predictions_LAMPE = np.zeros(len(cycles))
		predictions_LAMNE = np.zeros(len(cycles))
	
		for cycle in range(x_test.shape[1]):
				data = x_test[:, cycle, :]
				labels = y_test[:, cycle, :]
				predictions = model.predict(data)

				predictions_LLI[cycle] = rmspe(labels[:,0], predictions[:,0])
				predictions_LAMPE[cycle] = rmspe(labels[:,1], predictions[:,1])
				predictions_LAMNE[cycle] = rmspe(labels[:,2], predictions[:,2])
	
		df = pd.DataFrame(np.stack((predictions_LLI, predictions_LAMPE, predictions_LAMNE)), index=['LLI', 'LAMPE', 'LAMNE'],columns=[10, 50, 100, 200, 400, 1000])
		average.append(np.mean(df.mean(axis=1)))		
		display(df)

# falta por documentar en final
def get_data_eval(path, material, size, Q, cell_no, x_train_pre):
	'''
	Converts data to the format required by the models

	Parameters
	----------
	path: path where data is located
    material: str, chemistry of the battery
    size: int, size at which to reduce the IC data
    Q: array, capacity percentages from 0 to 100 from the simulated dataset
    cell_no: int, cell number to study
    x_train_pre: array, x_train needed to normalise data
	'''
	test_data = read_mat(path+'/x_test_'+cell_no+'.mat')['x_test'].T
	test_data = test_data.reshape(-1, test_data.shape[2]) # (n_samples, seq_len)
	test_data = convert_to_input_data(test_data, Q, size-1, material)
	test_data = normalise_data(test_data, np.min(x_train_pre), np.max(x_train_pre))
	return test_data

def plot_capacity_evolution(cycles, capacity_evolution, y_lim):
	'''
	Plots the capacity evolution of the battery.

	Parameters
	----------
	cycles: array, RPT measures of the cell
    capacity_evolution: array, capacity evolution of the cell
	y_lim: int, the limits of the y axis (percentage of capacity)
	'''
	
	plt.scatter(cycles, capacity_evolution, marker="^", s=100)
	plt.plot(cycles, capacity_evolution)
	plt.grid()
	plt.ylim(y_lim, 100)
	plt.xlabel('Cycle #')
	plt.ylabel('Normalized capacity (%)')
	plt.show()

def real_cells_to_percentage(Q, ci, ui, full_capacity, cycles):
	'''
	Converts the capacity of the real cells (given in Ah) to percentage.

	Parameters
	----------
	Q: array, capacity percentages from 0 to 100 from the simulated dataset
	ci: array, capacity values of the cell at each cycle
	ui: array, voltage values of the cell at each cycle
	full_capacity: float, the full capacity of the battery (Ah)
	cycles : array, RPT measures of the cell

	Returns
    -------
	ui_new: array, voltage values of the cell at each cycle in percentage
	capacity_evolution: array, the capacity evolution of the battery
    cycles: array, selected RPT measures of the cell
	'''
	
	ui_new = []
	capacity_evolution = []

	for i in range(len(ci)):
		# values must be in increasing order
		if np.all(np.diff(ci[i]) > 0):
			curve = pchip(ci[i], ui[i])
			# ci[i][-1]/full_capacity*100 gives the percentage of the current capacity
			# as it is a number with several decimals, it is rounded to 1 decimal, since this is the resolution of the variable Q
			current_capacity = np.around(ci[i][-1]/full_capacity*100,1)
			if current_capacity < 40:
				print('Degradation exceeds 40% from cycle '+str(cycles[i])+'. They are discarded as a consequence.')
				# that means the rest of the cycles also exceed 40%
				cycles = cycles[0:i]
				break
			capacity_evolution.append(current_capacity)
			
			# the value of Q that corresponds to that percentage is obtained and all the values of Q up to this value are taken (0, 0.1, ..., 94.2 e.g.)
			# the length of that portion of Q will be the length of the curve i
			len_curve = Q[0:np.where(Q==current_capacity)[0][0]+1].shape[0]
			# the curve is interpolated to the specified resolution (if not capacity is lost it will be interpolated to Q.shape[0] points)
			ci_augmented = np.linspace(min(ci[i]), max(ci[i]), len_curve)
			ui_augmented = curve(ci_augmented)
			# it means that it does not reach 100%, so it is filled with nans
			if ui_augmented.shape[0] != Q.shape[0]:
				for i in range(Q.shape[0]-ui_augmented.shape[0]):
					ui_augmented = np.append(ui_augmented, np.nan)
			ui_new.append(ui_augmented)
		else:
			print('Cycle '+str(cycles[i])+' is discarded as it is not increasingly sorted')
            # TO-DO: workaround, in this way another pop should not be done
			cycles.pop(i)
	return ui_new, capacity_evolution, cycles

def real_vs_simulated_curves(Q, V_reference_simulated, V_real, ui_reference_simulated, dqi_reference_simulated, size, material):
	'''
	Plots the reference voltage and IC curves for real and simulated curves

	Parameters
	----------
	Q: array, capacity percentages from 0 to 100 from the simulated dataset
	V_reference_simulated: array, voltage values of the reference simulated cell
	V_real: array, voltage values of the reference real cell
	ui_reference_simulated: array, voltage values of the reference simulated cell for the IC
	dqi_reference_simulated: float, IC values of the reference simulated cell
	size: int, the length of the curves
    material: str, chemistry of the battery
	'''

	plt.title('Reference voltage curve')
	plt.plot(Q, V_reference_simulated, label='Simulated')
	plt.ylim(3.2, 3.5)
	plt.plot(Q, V_real, label='Real')
	plt.legend()
	plt.show()

	# simulated
	IC_simulated = reduce_size(ui_reference_simulated, dqi_reference_simulated, size)
	IC_simulated = normalise_data(IC_simulated, np.min(IC_simulated), np.max(IC_simulated))

	# real
	min_v, max_v, path = get_minmaxV(material)
	ui_reference_real, dqi_reference_real = IC(V_real, Q, UI_STEP, min_v, max_v)
	IC_real = reduce_size(ui_reference_real, dqi_reference_real, size)
	IC_real = normalise_data(IC_real, np.min(IC_real), np.max(IC_real))

	plt.title('Reference IC')
	plt.plot(np.linspace(3.2, 3.5, len(IC_simulated)), IC_simulated, label='Simulated')
	plt.plot(np.linspace(3.2, 3.5, len(IC_real)), IC_real, label='Real')
	plt.legend()
	plt.show()

def convert_to_input_data(ui_new, Q, size, material):
	'''
	Converts the voltage values of the real cells to the input data for the neural network

	Parameters
	----------
	ui_new: array, voltage values of the cell at each cycle in percentage
	Q: array, capacity percentages from 0 to 100 from the simulated dataset
	size: int, the length of the curves
	material: str, chemistry of the cell

	Returns
	-------
	x_test: array, the input data for the neural network
	'''
	min_v, max_v, path = get_minmaxV(material)
	samples = []
	for sample in range(len(ui_new)):
		# convert to IC
		ui_sample, dqi_sample = IC(ui_new[sample], Q, UI_STEP, min_v, max_v)
		# reduce size
		new_sample = reduce_size(ui_sample, dqi_sample, size)
		samples.append(new_sample)
	x_test = np.array(samples)
	return x_test

def get_capacity_prediction(info, predictions):
	'''
	Gets capacity from predictions of the degradation modes

	Parameters
	----------
	info: dictionary, contains the information about the training set
	predictions: array, the predictions of the model

	Returns
	-------
	capacity_prediction: array, capacity predictions
	'''
	from scipy.spatial import KDTree

	capacity_prediction = []
	for prediction in predictions:
		path = KDTree(info[:,0:3])
		_, index = path.query(prediction*100, k=1)
		capacity_prediction.append(100-info[index][3])

	return capacity_prediction

def plot_predictions(cycles, predictions, capacity_evolution, capacity_prediction, y_lim):
	'''
	Plots the predictions of the model

	Parameters
	----------
	cycles: array, cycles of the real cell
    predictions: array, predictions of the model
    capacity_evolution: array, capacity evolution of the real cell
    capacity_prediction: array, capacity predictions
    y_lim: float, the y limit of the plot
	'''
	# Degradation modes
	plt.plot(cycles, predictions[:,0]*100, label='LLI')
	plt.plot(cycles, predictions[:,1]*100, label='LAMPE')
	plt.plot(cycles, predictions[:,2]*100, label='LAMNE')
	plt.xlabel('Cycle #')
	plt.ylabel('Predicted degradation (%)')
	plt.legend()
	plt.show()

	# Capacity
	plt.scatter(cycles, capacity_evolution)
	plt.plot(cycles, capacity_evolution)
	plt.scatter(cycles, capacity_prediction)
	plt.plot(cycles, capacity_prediction)
	plt.ylim(y_lim, 100)
	plt.xlabel('Cycle #')
	plt.ylabel('Normalized capacity (%)')
	plt.show()