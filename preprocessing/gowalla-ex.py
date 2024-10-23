import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
import pickle
from tqdm import tqdm
import json


# Assuming the split_dataset function is not needed since we're not splitting
# If it's needed for some reason, you can import it, but it should not be used here

def applyParallelPD(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    from joblib import Parallel, delayed
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return pd.concat(df_ls)

def _get_time(df):
    min_day = pd.to_datetime(df["started_at"].min().date())
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["weekday"] = df["started_at"].dt.weekday
    return df

def enrich_time_info(sp):
    tqdm.pandas(desc="Time enriching")
    sp = applyParallelPD(sp.groupby("user_id", group_keys=False), _get_time, n_jobs=-1, print_progress=True)
    sp.drop(columns={"started_at"}, inplace=True)
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp = sp.reset_index(drop=True)

    sp["location_id"] = sp["location_id"].astype(int)
    sp["user_id"] = sp["user_id"].astype(int)

    # Avoid naming conflict by renaming 'location_id' to 'loc_id' instead of 'id'
    # sp = sp.rename(columns={"location_id": "loc_id"})

    # Assign a unique 'id' column for indexing
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    
    return sp


def preprocess_excluded_dataset(config, excluded_users_file):
    # Load excluded user dataset
    excluded_data = pd.read_csv(
        excluded_users_file,
        sep="\t",
        header=None,
        parse_dates=[1],
        names=["user_id", "started_at", "latitude", "longitude", "location_id"]
    )

    # Enrich the time information
    excluded_enriched = enrich_time_info(excluded_data)

    # **New Step: Filter users with insufficient data**
    # Keep users that have at least 3 records for valid sequence generation
    user_size = excluded_enriched.groupby(["user_id"]).size()
    valid_users = user_size[user_size >= 3].index
    excluded_enriched = excluded_enriched.loc[excluded_enriched["user_id"].isin(valid_users)]

    print(f"Shape of the dataset before encoding: {excluded_enriched.shape}")

    # Ordinal encode locations
    enc = OrdinalEncoder(
        dtype=np.int64,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ).fit(excluded_enriched["location_id"].values.reshape(-1, 1))  # Correct the column name here
    excluded_enriched["location_id"] = enc.transform(excluded_enriched["location_id"].values.reshape(-1, 1)) + 2

    print(
        f"Max location id: {excluded_enriched.location_id.max()}, unique location id: {excluded_enriched.location_id.unique().shape[0]}"
    )

    # Normalize coordinates
    excluded_enriched["longitude"] = (
        2 * (excluded_enriched["longitude"] - excluded_enriched["longitude"].min()) /
        (excluded_enriched["longitude"].max() - excluded_enriched["longitude"].min()) - 1
    )
    excluded_enriched["latitude"] = (
        2 * (excluded_enriched["latitude"] - excluded_enriched["latitude"].min()) /
        (excluded_enriched["latitude"].max() - excluded_enriched["latitude"].min()) - 1
    )

    # Extract unique location information
    gowalla_loc = (
        excluded_enriched.groupby(["location_id"])
        .head(1)
        .drop(columns=["user_id", "start_day", "start_min", "weekday"])
    )
    gowalla_loc = gowalla_loc.rename(columns={"location_id": "id"})

    print(
        f"Max user id: {excluded_enriched.user_id.max()}, unique user id: {excluded_enriched.user_id.unique().shape[0]}"
    )

    # Save the preprocessed dataset and locations
    data_path = f"./data-ex/valid_ids_gowalla.pk"
    final_valid_id = excluded_enriched["id"].values
    with open(data_path, "wb") as handle:
        pickle.dump(final_valid_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    excluded_enriched.to_csv(f"./data-ex/dataset_gowalla.csv", index=False)
    gowalla_loc.to_csv(f"./data-ex/locations_gowalla.csv", index=False)

    print("Preprocessed excluded dataset saved:")
    print(f"- Valid IDs: {data_path}")
    print(f"- Dataset: ./data-ex/dataset_gowalla.csv")
    print(f"- Locations: ./data-ex/locations_gowalla.csv")
    print(f"Final user size: {excluded_enriched['user_id'].unique().shape[0]}")


def applyParallelPD(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    from joblib import Parallel, delayed
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return pd.concat(df_ls)

if __name__ == "__main__":
    CONFIG_PATH = "paths.json" 
    EXCLUDED_USERS_FILE = "./data-ex/gowalla/100-excluded-users.txt"  # Path to excluded users' file

    # Load the config file
    with open(CONFIG_PATH) as json_file:
        CONFIG = json.load(json_file)

    # Preprocess the excluded user data
    preprocess_excluded_dataset(config=CONFIG, excluded_users_file=EXCLUDED_USERS_FILE)
