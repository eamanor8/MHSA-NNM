import torch, os
import numpy as np
import argparse
import pandas as pd
from datetime import datetime
import json

import torch.distributed as dist
import torch.multiprocessing as mp

from easydict import EasyDict as edict

from utils.utils import load_config, setup_seed, get_test_result, get_dataloaders, get_models


setup_seed(42)


def test_only(test_loader, config, device, log_dir, model_path):
    result_ls = []

    # Load pre-trained model
    model = get_models(config, device)
    model.load_state_dict(torch.load(model_path, map_location=f"cuda:{device}"))
    model.to(device)

    # Test
    perf, test_df = get_test_result(config, model, test_loader, device)
    test_df.to_csv(os.path.join(log_dir, "user_detail.csv"))

    result_ls.append(perf)

    return result_ls


def init_save_path(config, time_now):
    """define the path to save, and save the configuration file."""
    networkName = f"{config.dataset}_{config.networkName}"
    log_dir = os.path.join(config.save_root, f"{networkName}_{config.previous_day}_{str(time_now)}")
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


def main(rank, world_size, config, time_now, model_path):
    # setup the process groups
    setup(rank, world_size)

    torch.cuda.set_device(rank)

    result_ls = []

    # Get only the test_loader
    _, _, test_loader = get_dataloaders(rank, world_size, config)

    # Save configuration path
    log_dir = init_save_path(config, time_now)

    # Run testing only
    res_single = test_only(test_loader, config, rank, log_dir, model_path)

    data = {"tensor": res_single}

    # Gather results across distributed processes (if used)
    outputs = [None for _ in range(world_size)]
    dist.all_gather_object(outputs, data)

    # Collect results at the master node
    if rank == 0:
        print(outputs[0]["tensor"])
        result_ls.extend(outputs[0]["tensor"])

    result_df = pd.DataFrame(result_ls)
    if rank == 0:
        print(result_df)

        # Save the test results
        filename = os.path.join(
            config.save_root,
            f"{config.dataset}_{config.networkName}_test_results_{str(int(datetime.now().timestamp()))}.csv",
        )
        result_df.to_csv(filename, index=False)

    cleanup()


if __name__ == "__main__":

    world_size = 1

    time_now = int(datetime.now().timestamp())
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", help="Config file path.", default="config/gowalla/transformer.yml")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config = edict(config)

    mp.spawn(
        main,
        args=(world_size, config, time_now, args.model_path),
        nprocs=world_size,
    )
