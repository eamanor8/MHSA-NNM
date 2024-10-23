import torch, os
import numpy as np
import argparse
import pandas as pd
from datetime import datetime
import json
import torch.distributed as dist
import torch.multiprocessing as mp

from easydict import EasyDict as edict
from utils.utils import load_config, setup_seed, get_test_result, get_models
from utils.dataloader import sp_loc_dataset, collate_fn
from torch.utils.data import DataLoader

# Set random seed
setup_seed(42)

def single_run(excluded_loader, config, device, log_dir, model_path):
    # Load pre-trained model
    model = get_models(config, device)
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{device}'))
    model.to(device)
    model.eval()

    # Test on excluded users
    performance, result_user_df = get_test_result(config, model, excluded_loader, device)
    result_user_df.to_csv(os.path.join(log_dir, "excluded_users_test_results.csv"))

    return performance

def init_save_path(config, time_now, i):
    """Define the path to save and save the configuration file."""
    networkName = f"{config.dataset}_{config.networkName}"
    log_dir = os.path.join(config.save_root, f"{networkName}_{config.previous_day}_{str(time_now)}_{str(i)}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    return log_dir

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_excluded_users(rank, world_size, config, time_now, excluded_users_file, model_path):
    # Setup the distributed environment
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Load the excluded dataset
    excluded_dataset = sp_loc_dataset(
        source_root=config.source_root,
        data_type="test",
        dataset=config.dataset,
        previous_day=config.previous_day
    )
    excluded_loader = DataLoader(excluded_dataset, collate_fn=collate_fn, batch_size=config.batch_size, shuffle=False)

    # Save path for logs
    log_dir = init_save_path(config, time_now, rank)

    # Perform the test
    performance = single_run(excluded_loader, config, rank, log_dir, model_path)

    if rank == 0:
        print(performance)
    
    cleanup()

if __name__ == "__main__":
    world_size = 1
    time_now = int(datetime.now().timestamp())

    # Load configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", help="Config file path.", default="config/gowalla/ex-transformer.yml")
    parser.add_argument("--excluded_users_file", type=str, help="File path for excluded users' data", default="./data-ex/gowalla/100-excluded-users.txt")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model", default="saved-models/checkpoint.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    config = edict(config)

    mp.spawn(
        test_excluded_users,
        args=(world_size, config, time_now, args.excluded_users_file, args.model_path),
        nprocs=world_size,
    )
