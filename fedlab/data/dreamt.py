import os
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing import cpu_count
from urllib.parse import urljoin

import pandas as pd
import requests
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
        raise RuntimeError(
            "Export DREAMT_USERNAME and DREAMT_PASSWORD as the credentials of physionet.org"
        )

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
    dst_path = src_path.replace(".csv", f"_{fs_out}Hz.csv")

    if os.path.exists(dst_path):
        print(f"Skipping {filename}, already exists")
        return

    os.makedirs("/".join(src_path.split("/")[:-1]), exist_ok=True)

    n_in = fs_in * win_size
    n_out = 1000000 // fs_out

    chunk_size = n_in * 100
    reader = pd.read_csv(src_path, chunksize=chunk_size)

    with open(dst_path, "w") as f_out:
        header_written = False

        for chunk in reader:
            chunk = chunk[
                [
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
            ]

            chunk["TIMESTAMP"] = pd.to_timedelta(chunk["TIMESTAMP"], unit="s")
            chunk = chunk.set_index("TIMESTAMP")

            chunk["Window_Id"] = (chunk.index.total_seconds() // win_size).astype(int)

            chunk = chunk.groupby("Window_Id").filter(
                lambda g: len(g) == n_in
                and g["Sleep_Stage"].nunique() == 1
                and g.notna().all().all()
            )

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

            chunk.drop("Window_Id", errors="ignore")

            chunk.to_csv(f_out, index=False, header=not header_written)
            header_written = True

    print(f"{filename} done!")


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
        preprocess(filename, 64, 16, 30)

    with ThreadPoolExecutor(numOfWorkers) as executor:
        features = {executor.submit(task, f): f for f in filenames}
        wait(features)

    df = pd.read_csv("data/dreamt/data_64Hz/S002_whole_df.csv")
    print(df.info())
    df = pd.read_csv("data/dreamt/data_64Hz/S002_whole_df_16Hz.csv")
    print(df.info())
