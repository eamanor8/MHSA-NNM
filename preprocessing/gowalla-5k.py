import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
import json

from joblib import Parallel, delayed

from tqdm import tqdm
import pickle as pickle

from utils import split_dataset


def applyParallelPD(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
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

def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return df_ls

def getValidSequence(input_df, previous_day):
    valid_user_ls = applyParallel(
        input_df.groupby("user_id"),
        getValidSequenceUser,
        previous_day=previous_day,
        n_jobs=-1,
    )
    return [item for sublist in valid_user_ls for item in sublist]

def getValidSequenceUser(df, previous_day):
    df.reset_index(drop=True, inplace=True)

    valid_id = []
    min_days = df["start_day"].min()
    df["diff_day"] = df["start_day"] - min_days

    for index, row in df.iterrows():
        if row["diff_day"] < previous_day:
            continue

        hist = df.iloc[:index]
        hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]

        if len(hist) < 3:
            continue
        valid_id.append(row["id"])

    return valid_id

def enrich_time_info(sp):
    tqdm.pandas(desc="Time enriching")
    sp = applyParallelPD(sp.groupby("user_id", group_keys=False), _get_time, n_jobs=-1, print_progress=True)
    sp.drop(columns={"started_at"}, inplace=True)
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp = sp.reset_index(drop=True)

    sp["location_id"] = sp["location_id"].astype(int)
    sp["user_id"] = sp["user_id"].astype(int)

    sp.index.name = "id"
    sp.reset_index(inplace=True)
    return sp


def get_dataset(config):
    gowalla = pd.read_csv(
        os.path.join(config[f"raw_gowalla"], "Gowalla_totalCheckins.txt"),
        sep="\t",
        header=None,
        parse_dates=[1],
        names=["user_id", "started_at", "latitude", "longitude", "location_id"],
    )

    gowalla_enriched = enrich_time_info(gowalla)

    user_size = gowalla_enriched.groupby(["user_id"]).size()
    valid_users = user_size[user_size > 10].index
    gowalla_enriched = gowalla_enriched.loc[gowalla_enriched["user_id"].isin(valid_users)]

    poi_size = gowalla_enriched.groupby(["location_id"]).size()
    valid_pois = poi_size[poi_size > 10].index
    gowalla_enriched = gowalla_enriched.loc[gowalla_enriched["location_id"].isin(valid_pois)]

    valid_users_list = gowalla_enriched["user_id"].unique()

    num_users = len(valid_users_list)
    split_idx = num_users // 2

    batch1_users = valid_users_list[:5000]
    batch2_users = valid_users_list[5000:10000]  # Adjust as needed

    gowalla_batch1 = gowalla_enriched[gowalla_enriched["user_id"].isin(batch1_users)]
    gowalla_batch2 = gowalla_enriched[gowalla_enriched["user_id"].isin(batch2_users)]

    gowalla_batch1, final_valid_id1 = process_batch(gowalla_batch1, config)
    gowalla_batch2, final_valid_id2 = process_batch(gowalla_batch2, config)

    save_dataset(gowalla_batch1, final_valid_id1, "batch1")
    save_dataset(gowalla_batch2, final_valid_id2, "batch2")

    print("Final user size for Batch 1: ", gowalla_batch1["user_id"].unique().shape[0])
    print("Final user size for Batch 2: ", gowalla_batch2["user_id"].unique().shape[0])

def process_batch(gowalla, config):
    train_data, vali_data, test_data = split_dataset(gowalla)

    enc = OrdinalEncoder(
        dtype=np.int64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).fit(train_data["location_id"].values.reshape(-1, 1))
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2

    print(
        f"Max location id:{train_data.location_id.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
    )

    valid_ids = getValidSequence(train_data, previous_day=7)
    valid_ids.extend(getValidSequence(vali_data, previous_day=7))
    valid_ids.extend(getValidSequence(test_data, previous_day=7))

    all_ids = gowalla[["id"]].copy()
    all_ids["7"] = 0
    all_ids.loc[all_ids["id"].isin(valid_ids), "7"] = 1

    final_valid_id = all_ids.loc[all_ids.sum(axis=1) == all_ids.shape[1]].reset_index()["id"].values

    print(
        f"Max user id:{gowalla.user_id.max()}, unique user id:{gowalla.user_id.unique().shape[0]}"
    )

    return gowalla, final_valid_id

def save_dataset(gowalla, valid_ids, batch_name):
    data_path = f"./data-5k/{batch_name}_valid_ids_gowalla.pk"
    with open(data_path, "wb") as handle:
        pickle.dump(valid_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    gowalla.to_csv(f"./data-5k/{batch_name}_dataset_gowalla.csv", index=False)
    gowalla_loc = (
        gowalla.groupby(["location_id"])
        .head(1)
        .drop(columns={"id", "user_id", "start_day", "start_min", "weekday"})
    )
    gowalla_loc = gowalla_loc.rename(columns={"location_id": "id"})
    gowalla_loc.to_csv(f"./data-5k/{batch_name}_locations_gowalla.csv", index=False)
    
    print("Final user size: ", gowalla["user_id"].unique().shape[0])

if __name__ == "__main__":
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    get_dataset(config=CONFIG)
