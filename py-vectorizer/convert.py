import torch
import numpy as np

# Load the state dict
state_dict = torch.load("model_weights.pth", map_location="cpu")

# Example: convert entire embedding layer weights to .npy
# Assuming your model had an embedding layer saved as 'embeddings.weight'
weights = state_dict["embeddings.weight"].cpu().numpy()
np.save("weights.npy", weights)
