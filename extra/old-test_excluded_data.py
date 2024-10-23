import torch, os
import numpy as np
import argparse
import pandas as pd
from datetime import datetime
import json

import torch.distributed as dist
import torch.multiprocessing as mp

from easydict import EasyDict as edict

from utils.utils import load_config, get_test_result, get_models
from utils.dataloader import sp_loc_dataset, collate_fn
from torch.utils.data import DataLoader


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def test_excluded_users(rank, world_size, config, time_now, excluded_users_file, model_path):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Load excluded users' data
    excluded_dataset = sp_loc_dataset(
        source_root=config.source_root,
        data_type="test",
        dataset=config.dataset,
        previous_day=config.previous_day
    )
    excluded_loader = DataLoader(excluded_dataset, collate_fn=collate_fn, batch_size=config.batch_size, shuffle=False)

    # Load the pre-trained model
    model = get_models(config, rank)
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{rank}'))
    model.to(rank)
    model.eval()

    # Test on excluded users' data
    performance, result_user_df = get_test_result(config, model, excluded_loader, rank)

    if rank == 0:
        log_dir = os.path.join(config.save_root, f"excluded_test_results_{str(time_now)}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        result_user_df.to_csv(os.path.join(log_dir, "excluded_users_test_results.csv"))
        print("Testing on excluded users completed.")

    cleanup()


if __name__ == "__main__":
    world_size = 1

    time_now = int(datetime.now().timestamp())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, nargs="?", help="Config file path.", default="config/gowalla/transformer.yml"
    )
    parser.add_argument(
        "--excluded_users_file", type=str, help="File path for excluded users' data", default="./data-ex/gowalla/100-excluded-users.txt"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the pre-trained model", default="saved-models/checkpoint.pt"
    )

    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)

    mp.spawn(
        test_excluded_users,
        args=(world_size, config, time_now, args.excluded_users_file, args.model_path),
        nprocs=world_size,
    )
