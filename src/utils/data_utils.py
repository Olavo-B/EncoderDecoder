import numpy as np

def load_data(file_path):
    # Implementation of loading data logic
    print(f"Loading data from {file_path}")
    return None

# Step 2: Generate Example Data
def generate_dummy_data(num_samples=1000, input_dim=10):
    """Generates dummy data for testing the dataloader."""
    return np.random.rand(num_samples, input_dim)

# Step 3: Transformations (Optional)
def normalize_data(sample):
    """Normalize the sample to have values between 0 and 1."""
    return sample / np.max(sample)