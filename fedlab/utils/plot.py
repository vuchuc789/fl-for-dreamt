import matplotlib.pyplot as plt


class LivePlot:
    def __init__(self):
        plt.ion()  # interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax2 = self.ax.twinx()  # secondary y-axis for accuracy

        self.epochs = []
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []

        (self.train_line,) = self.ax.plot([], [], label="Train Loss", color="tab:blue")
        (self.test_line,) = self.ax.plot([], [], label="Test Loss", color="tab:orange")
        (self.acc_line,) = self.ax2.plot([], [], label="Accuracy", color="tab:green")

        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax2.set_ylabel("Accuracy")
        self.ax2.set_ylim(0, 1)  # <-- Fix accuracy range to 0–1
        self.ax.grid(True)

        # Combine legends from both axes
        lines = [self.train_line, self.test_line, self.acc_line]
        labels = [line.get_label() for line in lines]
        self.ax.legend(
            lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3
        )

    def update(self, epoch, train_loss, test_loss=None, accuracy=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_line.set_data(self.epochs, self.train_losses)

        if test_loss is not None:
            self.test_losses.append(test_loss)
            self.test_line.set_data(self.epochs, self.test_losses)

        if accuracy is not None:
            self.accuracies.append(accuracy)
            self.acc_line.set_data(self.epochs, self.accuracies)

        # Rescale left axis dynamically (loss)
        self.ax.relim()
        self.ax.autoscale_view()

        # Right axis stays fixed at (0, 1)
        self.ax2.set_ylim(0, 1)

        # Draw updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def show(self):
        plt.ioff()
        plt.show()


def plot_signals(df):
    signals = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]

    # pick colors per label
    label_colors = {
        lbl: color
        for lbl, color in zip(
            df["Sleep_Stage"].unique(),
            plt.cm.tab10.colors[: df["Sleep_Stage"].nunique()],
        )
    }

    fig, axes = plt.subplots(len(signals), 1, figsize=(15, 12), sharex=True)

    for ax, sig in zip(axes, signals):
        for lbl, color in label_colors.items():
            sub_df = df[df["Sleep_Stage"] == lbl]
            ax.plot(
                sub_df["TIMESTAMP"],
                sub_df[sig],
                color=color,
                linewidth=0.7,
                alpha=0.8,
                label=f"Stage {lbl}",
            )
        ax.set_ylabel(sig)

    axes[-1].set_xlabel("Timestamp")
    axes[0].legend(loc="upper right", ncol=len(label_colors))
    plt.tight_layout()
    plt.show()
