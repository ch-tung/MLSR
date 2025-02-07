import numpy as np
from scipy.io import loadmat
from scipy import stats
import yaml
import os

def load_training_data(config_file, logA=True, extend=False):
    # Load configuration from YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract parameters from config
    grid_file = os.path.join(os.getcwd(), 'KAN', config["mat_filename_grid"])
    rand_file = os.path.join(os.getcwd(), 'KAN', config["mat_filename_rand"])
    index_parameters_expr = config["index_parameters"]
    exp_scale = config["exp_scale"]
    outliers_threshold = config["outliers_threshold"]
    
    # Load the data from the .mat files
    data_grid = loadmat(grid_file)
    data_rand = loadmat(rand_file)
    
    # Extract the parameters and SQ lists
    parameters_list_grid = data_grid["parameters_list"]
    SQ_list_grid = data_grid["SQ_list"][:, 5:]
    parameters_list_rand = data_rand["parameters_list"]
    SQ_list_rand = data_rand["SQ_list"][:, 5:]
    
    parameters_list = parameters_list_rand
    SQ_list = SQ_list_rand
    
    # Exclude highest interaction
    index_parameters = eval(index_parameters_expr)
    parameters_list = parameters_list[index_parameters, :]
    SQ_list = SQ_list[index_parameters, :]
    
    # Rescale the training set SQ to range [0,1]
    def f_inp(sq):
        return np.log(sq) / exp_scale / 2

    # Transform the decoder output to SQ
    def f_out(sq_pred):
        return np.exp((sq_pred * 2) * exp_scale)  # inverse of f_inp

    def f_out_torch(sq_pred):
        return torch.exp((sq_pred * 2) * exp_scale)  # inverse of f_inp

    parameters_mean = np.mean(parameters_list, axis=0)
    parameters_std = np.std(parameters_list, axis=0)
    print(f"parameters mean: {parameters_mean}")
    print(f"parameters std: {parameters_std}")

    y_train = SQ_list
    x_train = np.array([(parameters_list[:, i]) for i in range(3)]).T
    
    index_ext = ((parameters_list[:,0]>0.21)*(parameters_list[:,1]<0.1))
    
    y_train_ext = y_train[index_ext,:]
    x_train_ext = np.array([(parameters_list[index_ext, i]) for i in range(3)]).T

    # y_train_ext = SQ_list_grid
    # x_train_ext = np.array([(parameters_list_grid[:, i]) for i in range(3)]).T

    # Extend training set
    if extend:
        y_train = np.vstack((y_train, y_train_ext))
        x_train = np.vstack((x_train, x_train_ext))
        
    print(f"parameters shape: {x_train.shape}")
    print(f"SQ shape: {y_train.shape}")

    def f_params_z(parameters):
        return np.array([(parameters[i] - parameters_mean[i]) / parameters_std[i] for i in range(3)])

    Q_train = np.linspace(1.2, 20, 95)
    print(f"Q shape: {Q_train.shape}")

    # Identify outliers
    y_train_centered = y_train - np.mean(y_train, axis=0)
    U, S, Vt = np.linalg.svd(y_train_centered, full_matrices=False)

    # Get the first 3 principal components
    PC = U[:, :3]

    # Compute the Z-scores of the principal components
    z_scores = np.abs(stats.zscore(PC))

    outliers = np.where(z_scores > outliers_threshold)  # use threshold from config

    # Print the indices of the outliers
    print("Outliers are at indices:", outliers)

    # Remove outliers
    y_train = np.delete(y_train, outliers, axis=0)
    x_train = np.delete(x_train, outliers, axis=0)
    
    # Take log value on the third column of x_train
    if logA:
        x_train[:, 2] = np.log(x_train[:, 2])

    return x_train, y_train, Q_train

def load_training_data_grid(config_file, logA=True, extend=False):
    # Load configuration from YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract parameters from config
    grid_file = config["mat_filename_grid"]
    rand_file = config["mat_filename_rand"]
    index_parameters_expr = config["index_parameters"]
    exp_scale = config["exp_scale"]
    outliers_threshold = config["outliers_threshold"]
    
    # Load the data from the .mat files
    data_grid = loadmat(grid_file)
    data_rand = loadmat(rand_file)
    
    # Extract the parameters and SQ lists
    parameters_list_grid = data_grid["parameters_list"]
    SQ_list_grid = data_grid["SQ_list"][:, 5:]
    
    # Exclude highest interaction
    index_parameters = eval(index_parameters_expr)
    parameters_list = parameters_list[index_parameters, :]
    SQ_list = SQ_list[index_parameters, :]
    
    print(f"parameters shape: {parameters_list.shape}")
    print(f"SQ shape: {SQ_list.shape}")
    
    # Rescale the training set SQ to range [0,1]
    def f_inp(sq):
        return np.log(sq) / exp_scale / 2

    # Transform the decoder output to SQ
    def f_out(sq_pred):
        return np.exp((sq_pred * 2) * exp_scale)  # inverse of f_inp

    def f_out_torch(sq_pred):
        return torch.exp((sq_pred * 2) * exp_scale)  # inverse of f_inp

    parameters_mean = np.mean(parameters_list, axis=0)
    parameters_std = np.std(parameters_list, axis=0)
    print(f"parameters mean: {parameters_mean}")
    print(f"parameters std: {parameters_std}")

    y_train = SQ_list_grid
    x_train = np.array([(parameters_list_grid[:, i]) for i in range(3)]).T

    def f_params_z(parameters):
        return np.array([(parameters[i] - parameters_mean[i]) / parameters_std[i] for i in range(3)])

    Q_train = np.linspace(1.2, 20, 95)
    print(f"Q shape: {Q_train.shape}")

    # Identify outliers
    y_train_centered = y_train - np.mean(y_train, axis=0)
    U, S, Vt = np.linalg.svd(y_train_centered, full_matrices=False)

    # Get the first 3 principal components
    PC = U[:, :3]

    # Compute the Z-scores of the principal components
    z_scores = np.abs(stats.zscore(PC))

    outliers = np.where(z_scores > outliers_threshold)  # use threshold from config

    # Print the indices of the outliers
    print("Outliers are at indices:", outliers)

    # Remove outliers
    y_train = np.delete(y_train, outliers, axis=0)
    x_train = np.delete(x_train, outliers, axis=0)
    
    # Take log value on the third column of x_train
    if logA:
        x_train[:, 2] = np.log(x_train[:, 2])

    return x_train, y_train, Q_train

def load_grid_data(config_file):
    # Load configuration from YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract parameters from config
    grid_file = config["mat_filename_grid"]
    rand_file = config["mat_filename_rand"]
    index_parameters_expr = config["index_parameters"]
    exp_scale = config["exp_scale"]
    outliers_threshold = config["outliers_threshold"]
    
    # Load the data from the .mat files
    data_grid = loadmat(grid_file)
    
    # Extract the parameters and SQ lists
    parameters_list = data_grid["parameters_list"]
    SQ_list = data_grid["SQ_list"][:, 5:]
    
    return parameters_list, SQ_list

# # Example usage
# config_file = 'config.yaml'
# x_train, y_train, Q_train = load_training_data(config_file)
