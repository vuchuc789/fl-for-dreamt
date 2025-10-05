import os
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from urllib.parse import urljoin

import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

FREQUENCY_IN = 64  # Hz
FREQUENCY_OUT = 4  # Hz
WINDOW_SIZE = 30  # s


def download(filename: str):
    local_dir = "data/dreamt/"
    local_path = os.path.join(local_dir, filename)

    # Skip if already downloaded
    if os.path.exists(local_path):
        print(f"[{filename}] ⚙️ Skipped — already exists.")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    username = os.getenv("DREAMT_USERNAME")
    password = os.getenv("DREAMT_PASSWORD")
    if not username or not password:
        print(
            f"[{filename}] ❌ Missing credentials. Please export DREAMT_USERNAME and DREAMT_PASSWORD."
        )
        return

    remote_dir = "https://physionet.org/files/dreamt/2.1.0/"
    remote_path = urljoin(remote_dir, filename)

    start_time = datetime.now()
    try:
        session = requests.Session()
        session.auth = (username, password)
        session.headers.update({"User-Agent": "Wget/1.25.0"})

        with session.get(remote_path, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            with (
                open(local_path, "wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"↓ {filename}",
                    initial=0,
                ) as bar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        elapsed = (datetime.now() - start_time).total_seconds()
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[{filename}] ✅ Downloaded {size_mb:.2f} MB in {elapsed:.1f}s.")

    except requests.exceptions.HTTPError as e:
        print(f"[{filename}] ❌ HTTP error: {e}")
    except requests.exceptions.ConnectionError:
        print(f"[{filename}] ❌ Connection error.")
    except Exception as e:
        print(f"[{filename}] ❌ Failed: {e}")


def preprocess(filename: str, fs_in: int, fs_out: int, win_size: int):
    dir_path = "data/dreamt/"
    src_path = os.path.join(dir_path, filename)
    dst_path = src_path.replace(".csv", "_preprocessed.csv")

    os.makedirs(os.path.dirname(src_path), exist_ok=True)

    n_in = fs_in * win_size
    n_out = 1000000 // fs_out  # microseconds step for resample
    chunk_size = n_in * 100

    total = remain = update = sample = 0
    start_time = datetime.now()

    try:
        reader = pd.read_csv(src_path, chunksize=chunk_size)
        with open(dst_path, "w") as f_out:
            header_written = False
            for chunk in reader:
                total += len(chunk)

                # Keep only required columns
                keep_cols = [
                    "TIMESTAMP",
                    "C4-M1",
                    "F4-M1",
                    "O2-M1",
                    "Fp1-O2",
                    "T3 - CZ",
                    "CZ - T4",
                    "CHIN",
                    "E1",
                    "E2",
                    "ECG",
                    "LAT",
                    "RAT",
                    "SNORE",
                    "PTAF",
                    "FLOW",
                    "THORAX",
                    "ABDOMEN",
                    "SAO2",
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
                chunk = chunk[[c for c in keep_cols if c in chunk.columns]]

                # TIMESTAMP → timedelta index
                chunk["TIMESTAMP"] = pd.to_timedelta(chunk["TIMESTAMP"], unit="s")
                chunk = chunk.set_index("TIMESTAMP")

                # Filter by window validity
                chunk["Window_Id"] = (chunk.index.total_seconds() // win_size).astype(
                    int
                )
                chunk = chunk.groupby("Window_Id").filter(
                    lambda g: (len(g) == n_in and g.notna().all().all())
                )
                remain += len(chunk)

                # Harmonize labels
                labels = chunk.groupby("Window_Id")["Sleep_Stage"].apply(
                    lambda x: x.mode().iloc[0]
                )
                original_labels = chunk["Sleep_Stage"]
                chunk["Sleep_Stage"] = chunk["Window_Id"].map(labels)
                update += (original_labels != chunk["Sleep_Stage"]).sum()

                # Resample if needed
                if fs_out < fs_in:
                    chunk = chunk.resample(f"{n_out}us").agg(
                        {
                            col: "mean"
                            for col in chunk.columns
                            if col not in ["Sleep_Stage", "Window_Id"]
                        }
                        | {"Sleep_Stage": "first"}
                    )
                sample += len(chunk)

                # Save
                chunk.drop(columns=["Window_Id"], inplace=True, errors="ignore")
                chunk.reset_index(inplace=True)
                chunk["TIMESTAMP"] = chunk["TIMESTAMP"].dt.total_seconds()
                chunk.to_csv(f_out, index=False, header=not header_written)
                header_written = True

        # Compute stats
        dropped_percent = ((total - remain) / total) * 100 if total else 0
        updated_percent = (update / remain) * 100 if remain else 0
        duration = (datetime.now() - start_time).total_seconds()

        print(
            f"[{filename}] ✅ Done | {total:,} rows ({fs_in}Hz) | "
            f"Dropped {total - remain:,} rows ({dropped_percent:.2f}%) | Updated {update:,} labels ({updated_percent:.2f}%) | "
            f"Resampled to {sample:,} rows ({fs_out}Hz) | Time: {duration:.1f}s"
        )

    except Exception as e:
        print(f"[{filename}] ❌ Failed: {e}")


class DREAMTDataset(Dataset):
    def __init__(
        self,
        participant: int,
        test=False,
        transform=None,
        target_transform=None,
    ):
        df = pd.read_csv(
            f"data/dreamt/data_100Hz/S{participant + 2:03d}_PSG_df_updated.csv"
            if FREQUENCY_IN == 100
            else f"data/dreamt/data_64Hz/S{participant + 2:03d}_whole_df_preprocessed.csv"
        )
        df.drop(columns=["TIMESTAMP"], inplace=True)

        X = df.drop(columns=["Sleep_Stage"]).to_numpy()
        y = df[["Sleep_Stage"]].to_numpy()

        feat_num = X.shape[1]
        win_size = WINDOW_SIZE * FREQUENCY_OUT  # number of time steps per window

        n = len(X) - len(X) % win_size  # to trim unfitted time steps
        X = X[:n].reshape(-1, win_size, feat_num)  # (win_num, win_size, feat_num)
        y = y[:n].reshape(-1, win_size)[:, 0]  # all same per window

        # split by windows not steps
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, feat_num)  # (win_num * win_size, feat_num)
        scaler.fit(X_train_flat)

        X_flat = (X_test if test else X_train).reshape(
            -1, feat_num
        )  # (win_num * win_size, feat_num)

        X_scaled = scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(
            -1, win_size, feat_num
        )  # (win_num, win_size, feat_num)

        self.X = X_scaled.astype("float32")
        self.y = (
            LabelEncoder()
            .fit(["P", "W", "N1", "N2", "N3", "R"])
            .transform(y_test if test else y_train)
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]  # (win_size, feat_num) or (seq, feature)
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)

        return X, y


def load_data(participant: int, batch_size: int = 32):
    trainloader = DataLoader(
        DREAMTDataset(participant, test=False, transform=from_numpy),
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        DREAMTDataset(participant, test=True, transform=from_numpy),
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, testloader


if __name__ == "__main__":
    # load_data(0)
    # rnn = nn.GRU(8, 128, 1, batch_first=True)
    # fc = nn.Linear(128, 6)
    # for input, _ in load_data(0)[0]:
    #     output, h_n = rnn(input)
    #     out = fc(h_n[-1])
    #     print(input.shape)
    #     print(output.shape)
    #     print(h_n.shape)
    #     print(out.shape)
    #     break

    numOfParticipants = 4
    numOfWorkers = os.cpu_count() // 2

    filenames = [
        (
            f"data_100Hz/S{i:03d}_PSG_df_updated.csv"
            if FREQUENCY_IN == 100
            else f"data_64Hz/S{i:03d}_whole_df.csv"
        )
        for i in range(2, numOfParticipants + 2)
    ]

    def task(filename: str):
        download(filename)
        preprocess(filename, FREQUENCY_IN, FREQUENCY_OUT, WINDOW_SIZE)

    with ThreadPoolExecutor(numOfWorkers) as executor:
        features = {executor.submit(task, f): f for f in filenames}
        wait(features)

    df = pd.read_csv("data/dreamt/data_64Hz/S005_whole_df_preprocessed.csv")
    print(df.info())

    signals = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]

    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 8), sharex=True)

    # color map for labels
    label_colors = {
        lbl: col for lbl, col in zip(df["Sleep_Stage"].unique(), plt.cm.tab10.colors)
    }

    for i, sig in enumerate(signals):
        ax = axes[i]
        for lbl, color in label_colors.items():
            sub_df = df[df["Sleep_Stage"] == lbl]
            ax.scatter(
                sub_df["TIMESTAMP"],
                sub_df[sig],
                color=color,
                s=0.1,
                alpha=0.7,
                label=f"Label {lbl}" if i == 0 else "",
            )
        ax.set_ylabel(sig)

    axes[0].legend(
        loc="upper center",  # place legend above plots
        bbox_to_anchor=(0.5, 1.7),  # (x, y) relative to axes: 0.5 = center, 1.7 = above
        ncol=len(label_colors),  # all labels in one row
        frameon=True,
        markerscale=10,
    )
    plt.xlabel("Time (s)")
    plt.suptitle("Signals over Time", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    fig_w = len(signals) // 2
    fig, axes = plt.subplots(2, fig_w, figsize=(12, 8), tight_layout=True)
    for i, sig in enumerate(signals):
        axes[i // fig_w, i % fig_w].hist(df[sig])
        axes[i // fig_w, i % fig_w].set_title(sig)

    plt.suptitle("Signal Distributions", fontsize=16)
    plt.show()
