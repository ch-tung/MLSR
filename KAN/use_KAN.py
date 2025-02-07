import torch
import torch.nn as nn
import yaml
from kan import *
from use_training_set import *
import os

# Default device
device = torch.device('cpu')
torch.set_default_dtype(torch.float32)

def set_device(new_device):
    global device
    device = torch.device(new_device)
    print(f"Device set to: {device}")
    
def update_device(new_device):
    global device, x_train_torch, y_train_torch
    set_device(new_device)
    x_train_torch = x_train_torch.to(device)
    y_train_torch = y_train_torch.to(device)
    print("All relevant tensors and models have been moved to the new device.")

def calculate_output_size(input_size, filter_size, padding, stride):
    return int((input_size - filter_size + 2*padding) / stride + 1)

def to_torch(array):
    return torch.from_numpy(array).float()

def to_torch_device(array, device=device):
    return torch.from_numpy(array.astype('float32')).float().to(device)

# Load training data
config_file = os.getcwd()+'/KAN/setup_ts_full.txt'
x_train, y_train, Q_train = load_training_data(config_file)
x_train_torch = to_torch_device(x_train, device=device)
y_train_torch = to_torch_device(y_train, device=device)

## define KAN model
Q = np.linspace(1.2,20,95)
class SQ_KAN(nn.Module):
    def __init__(self, width=[4,11,3], width_aug=[3,7,7,3], grid=10, grid_aug=10, k=3, seed=42, device='cpu', multiplier=40):
        super(SQ_KAN, self).__init__()
        self.kan_aug = KAN(width=width_aug, grid=grid_aug, k=k, seed=seed, device=device, noise_scale=.01, base_fun='identity')
        self.kan = KAN(width=width, grid=grid, k=k, seed=seed, device=device, noise_scale=.01, base_fun='identity')
        self.kan_aug.update_grid_from_samples(x_train_torch)
        self.Q_torch_scale = to_torch_device((Q-6)/20)
        self.Q_torch = to_torch_device(Q)
        self.multiplier = multiplier
        
    def forward(self, x):
        phi = x[:, 0]*1
        x = self.kan_aug(x)
        x_expanded = x.unsqueeze(1).expand(-1, self.Q_torch_scale.size(0), -1)
        Q_expanded = self.Q_torch_scale.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(-1))
        Q_params = torch.cat([Q_expanded, x_expanded], dim=-1)
        Q_params_reshaped = Q_params.view(-1, Q_params.size(-1))
        
        G_full = self.kan(Q_params_reshaped)
        G_full_reshaped = G_full.view(x.size(0), self.Q_torch_scale.size(0), 3)  # (n_sample, n_Q, 3)
        
        output_1 = G_full_reshaped[:, :, 0]
        output_2 = G_full_reshaped[:, :, 1]
        output_3 = G_full_reshaped[:, :, 2]
        
        G_HS_bias = (self.multiplier*output_1 * torch.sin(output_2))/self.Q_torch
        
        phi_expanded = phi.unsqueeze(1).expand_as(output_1)
        
        # Compute alpha, beta, and gama
        alpha = (1 + 2 * phi)**2 / (1 - phi)**4
        beta = -6 * phi * (1 + phi / 2)**2 / (1 - phi)**4
        gama = phi * alpha / 2
        
        # Compute G_hs(Q, phi)
        Q_torch = self.Q_torch.unsqueeze(0).expand(x.size(0), -1)
        alpha_expanded = alpha.unsqueeze(1).expand_as(Q_torch)
        beta_expanded = beta.unsqueeze(1).expand_as(Q_torch)
        gama_expanded = gama.unsqueeze(1).expand_as(Q_torch)
        
        G_hs = (alpha_expanded * (torch.sin(Q_torch) - Q_torch * torch.cos(Q_torch)) / Q_torch**2 +
                beta_expanded * (2 * Q_torch * torch.sin(Q_torch) + (2 - Q_torch**2) * torch.cos(Q_torch) - 2) / Q_torch**3 +
                gama_expanded * (-Q_torch**4 * torch.cos(Q_torch) + 4 * ((3 * Q_torch**2 - 6) * torch.cos(Q_torch) +
                (Q_torch**3 - 6 * Q_torch) * torch.sin(Q_torch) + 6)) / Q_torch**5)
        
        # Ensure G_hs has the shape (n_sample, n_Q)
        G_hs = G_hs.view(x.size(0), self.Q_torch.size(0))
        
        return 1 / (24 * phi_expanded * (G_hs + G_HS_bias) / self.Q_torch + 1 + output_3)
    
# Define the build_model function
def build_model(config, device='cuda'):
    model = SQ_KAN(
        width=config['width'],
        width_aug=config['width_aug'],
        grid=config['grid'],
        grid_aug=config['grid_aug'],
        k=config['k'],
        seed=config['seed'],
        device=device,
        multiplier=config['multiplier']
    )
    return model
    
def main():
    with open('setup_model.txt', 'r') as file:
        config = yaml.safe_load(file)
    
    model = build_model(config['Model Setup'])
    print(model)

if __name__ == "__main__":
    main()
