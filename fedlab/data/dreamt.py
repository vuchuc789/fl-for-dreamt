import os
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing import cpu_count
from urllib.parse import urljoin

import pandas as pd
import requests
from torch.utils.data import Dataset
from tqdm import tqdm


def download(filename: str):
    local_dir = "data/dreamt/"
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        print(f"Skipping {filename}, already exists")
        return

    os.makedirs("/".join(local_path.split("/")[:-1]), exist_ok=True)

    username = os.getenv("DREAMT_USERNAME")
    password = os.getenv("DREAMT_PASSWORD")

    if not username or not password:
        print(
            "Export DREAMT_USERNAME and DREAMT_PASSWORD as the credentials of physionet.org"
        )
        return

    remote_dir = "https://physionet.org/files/dreamt/2.1.0/"
    remote_path = urljoin(remote_dir, filename)

    session = requests.Session()
    session.auth = (username, password)  # Basic Auth
    session.headers.update({"User-Agent": "Wget/1.25.0"})  # pretend to be wget

    with session.get(remote_path, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with (
            open(local_path, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                initial=0,
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def preprocess(filename: str, fs_in: int, fs_out: int, win_size: int):
    dir_path = "data/dreamt/"
    src_path = os.path.join(dir_path, filename)
    dst_path = src_path.replace(".csv", "_preprocessed.csv")

    os.makedirs("/".join(src_path.split("/")[:-1]), exist_ok=True)

    n_in = fs_in * win_size
    n_out = 1000000 // fs_out  # microseconds step for resample

    chunk_size = n_in * 100
    reader = pd.read_csv(src_path, chunksize=chunk_size)

    with open(dst_path, "w") as f_out:
        header_written = False

        for chunk in reader:
            # keep only required columns
            chunk.drop(
                columns=[
                    c
                    for c in chunk.columns
                    if c
                    not in [
                        "TIMESTAMP",
                        "BVP",
                        "ACC_X",
                        "ACC_Y",
                        "ACC_Z",
                        "TEMP",
                        "EDA",
                        "HR",
                        "IBI",
                        "Sleep_Stage",
                    ]
                ],
                inplace=True,
            )

            # TIMESTAMP → timedelta index
            chunk["TIMESTAMP"] = pd.to_timedelta(chunk["TIMESTAMP"], unit="s")
            chunk = chunk.set_index("TIMESTAMP")

            # assign window IDs
            chunk["Window_Id"] = (chunk.index.total_seconds() // win_size).astype(int)

            # filter + majority rule
            def valid_window(g):
                if len(g) != n_in or g.isna().any().any():
                    return False
                counts = g["Sleep_Stage"].value_counts()
                majority_frac = counts.iloc[0] / len(g)
                return majority_frac >= 0.8

            filtered = []
            for _, g in chunk.groupby("Window_Id"):
                if valid_window(g):
                    # force label = majority label
                    majority_label = g["Sleep_Stage"].mode().iloc[0]
                    g["Sleep_Stage"] = majority_label
                    filtered.append(g)

            if not filtered:
                continue
            chunk = pd.concat(filtered)

            # resample if needed
            if fs_out < fs_in:
                chunk = chunk.resample(f"{n_out}us").agg(
                    {
                        "BVP": "mean",
                        "ACC_X": "mean",
                        "ACC_Y": "mean",
                        "ACC_Z": "mean",
                        "TEMP": "mean",
                        "EDA": "mean",
                        "HR": "mean",
                        "IBI": "mean",
                        "Sleep_Stage": "first",
                    }
                )

            chunk.drop(columns=["Window_Id"], errors="ignore", inplace=True)
            chunk.reset_index(inplace=True)
            chunk["TIMESTAMP"] = chunk["TIMESTAMP"].dt.total_seconds()

            # save chunk
            chunk.to_csv(f_out, index=False, header=not header_written)
            header_written = True

    print(f"{filename} done!")


class DREAMTDataset(Dataset):
    pass


def load_data():
    pass


if __name__ == "__main__":
    numOfParticipants = 1
    numOfWorkers = cpu_count() // 2

    filenames = [
        f"data_64Hz/S{i:03d}_whole_df.csv" for i in range(2, numOfParticipants + 2)
    ]

    def task(filename: str):
        download(filename)
        preprocess(filename, 64, 4, 30)

    with ThreadPoolExecutor(numOfWorkers) as executor:
        features = {executor.submit(task, f): f for f in filenames}
        wait(features)

    # df = pd.read_csv("data/dreamt/data_64Hz/S002_whole_df_preprocessed.csv")
    # print(df.info())
    #
    # signals = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]
    #
    # fig, axes = plt.subplots(len(signals), 1, figsize=(12, 8), sharex=True)
    #
    # # color map for labels
    # label_colors = {
    #     lbl: col for lbl, col in zip(df["Sleep_Stage"].unique(), plt.cm.tab10.colors)
    # }
    #
    # for i, sig in enumerate(signals):
    #     ax = axes[i]
    #     for lbl, color in label_colors.items():
    #         sub_df = df[df["Sleep_Stage"] == lbl]
    #         ax.scatter(
    #             sub_df["TIMESTAMP"],
    #             sub_df[sig],
    #             color=color,
    #             s=0.1,
    #             alpha=0.7,
    #             label=f"Label {lbl}" if i == 0 else "",
    #         )
    #     ax.set_ylabel(sig)
    #
    # axes[0].legend(
    #     loc="upper center",  # place legend above plots
    #     bbox_to_anchor=(0.5, 1.7),  # (x, y) relative to axes: 0.5 = center, 1.7 = above
    #     ncol=len(label_colors),  # all labels in one row
    #     frameon=True,
    #     markerscale=10,
    # )
    # plt.xlabel("Time (s)")
    # plt.suptitle("Signals over Time", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    # plt.show()
