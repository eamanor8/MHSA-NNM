import os
import torch
import pandas as pd
from utils.utils import load_config, get_models
from utils.dataloader import sp_loc_dataset, collate_fn
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

# Import the complete test function from train.py
from utils.train import test

def test_on_excluded_users(config, excluded_users_file, model_path, device):
    # Load excluded users' data
    excluded_users_data = pd.read_csv(excluded_users_file, sep='\t', header=None)
    excluded_users_data.columns = ["user_id", "started_at", "latitude", "longitude", "location_id"]
    
    # Process the data as done in the `get_dataloaders` function
    excluded_dataset = sp_loc_dataset(
        source_root=config.source_root,
        data_type="test",  # treat excluded data as a test set
        dataset=config.dataset,
        previous_day=config.previous_day
    )

    excluded_loader = DataLoader(excluded_dataset, collate_fn=collate_fn, batch_size=config.batch_size, shuffle=False)

    # Load the trained model
    model = get_models(config, device)
    model.load_state_dict(torch.load(model_path))

    # Move the model to the device (GPU or CPU)
    model.to(device)

    # Call the existing test function to test the excluded users' data
    performance, result_dict = test(config, model, excluded_loader, device)

    # The test function already returns performance and prints it
    # Save results as needed
    results_path = os.path.join(config.save_root, "excluded_users_test_results.csv")
    pd.DataFrame([performance]).to_csv(results_path)

    print("Testing on excluded users completed.")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    # Load config
    config_path = "config/gowalla/transformer.yml"  # Path to the config file
    excluded_users_file = "./data/gowalla/100-excluded-users.txt"  # Path to excluded users' file
    model_path = "saved-models/checkpoint.pt"  # Path to the saved model after training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config = load_config(config_path)
    config = edict(config)

    # Run test on excluded users using the test function from train.py
    test_on_excluded_users(config, excluded_users_file, model_path, device)
