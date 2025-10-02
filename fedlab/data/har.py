from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset


def load_raw(test=False):
    paths = (
        (
            "UCI HAR Dataset/test/Inertial Signals/",
            "UCI HAR Dataset/test/y_test.txt",
        )
        if test
        else (
            "UCI HAR Dataset/train/Inertial Signals/",
            "UCI HAR Dataset/train/y_train.txt",
        )
    )

    X_keys = []
    X_df = pd.DataFrame()
    y_df = pd.DataFrame()

    with ZipFile("data/har.zip") as z:
        rawzip = z.open("UCI HAR Dataset.zip")
        with ZipFile(rawzip) as z:
            X_files = [
                f.filename
                for f in z.filelist
                if f.filename.startswith(paths[0]) and f.filename.endswith(".txt")
            ]
            X_keys = [
                "_".join(f.split("/")[-1].split(".")[0].split("_")[:-1])
                for f in X_files
            ]
            X_df = pd.concat(
                [pd.read_csv(z.open(f), header=None, sep="\\s+") for f in X_files],
                axis=1,
                keys=X_keys,
            )
            y_df = pd.read_csv(
                z.open(paths[1]),
                header=None,
                sep="\\s+",
                names=["label"],
            )

    return X_df, y_df


scaler = None


class HARDataset(Dataset):
    def __init__(
        self,
        test=False,
        transform=None,
        target_transform=None,
    ):
        self.X_df, self.y_df = load_raw(test)

        global scaler
        if scaler is None:
            scaler = {}

            for key in self.X_df.columns.levels[0]:
                scaler[key] = (
                    self.X_df[key].values.mean(),
                    self.X_df[key].values.std(),
                )

        # z-score normalization
        for key in self.X_df.columns.levels[0]:
            self.X_df[key] = (self.X_df[key] - scaler[key][0]) / scaler[key][1]

        self.X_df = self.X_df.astype("float32")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx):
        X = self.X_df.iloc[idx].values.reshape(-1, 128)  # (9, 128)
        y = self.y_df.iloc[idx, 0]

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)

        return X, y


def load_data(batch_size: int = 32):
    trainloader = DataLoader(
        HARDataset(test=False, transform=from_numpy),
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        HARDataset(test=True, transform=from_numpy),
        batch_size=batch_size,
        shuffle=False,
    )
    return trainloader, testloader


if __name__ == "__main__":
    df, _ = load_raw()

    fig, axs = plt.subplots(3, 3, figsize=(8, 8), tight_layout=True)

    for i, key in enumerate(df.columns.levels[0]):
        data = df[key].values.flatten()

        # data = (data - data.mean()) / data.std()

        axs[i // 3, i % 3].hist(data)
        axs[i // 3, i % 3].set_title(key)

        print(key, data.mean(), data.std(), data.min(), data.max())

    plt.show()
