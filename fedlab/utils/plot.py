import matplotlib.pyplot as plt


class LivePlot:
    def __init__(
        self,
        epochs: list[int] = [],
        train_losses: list[float] = [],
        test_losses: list[float] = [],
        metrics: list[float] = [],
        metric_name: str = "Accuracy",
    ):
        plt.ion()  # interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax2 = self.ax.twinx()  # secondary y-axis for metric

        self.epochs = epochs
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.metrics = metrics

        (self.train_line,) = self.ax.plot(
            epochs, train_losses, label="Train Loss", color="tab:blue"
        )
        (self.test_line,) = self.ax.plot(
            epochs, test_losses, label="Test Loss", color="tab:orange"
        )
        (self.acc_line,) = self.ax2.plot(
            epochs, metrics, label=metric_name, color="tab:green"
        )

        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax2.set_ylabel(metric_name)
        self.ax2.set_ylim(0, 1)  # <-- Fix metric range to 0–1
        self.ax.grid(True)

        # Combine legends from both axes
        lines = [self.train_line, self.test_line, self.acc_line]
        labels = [line.get_label() for line in lines]
        self.ax.legend(
            lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3
        )

    def update(self, epoch, train_loss=None, test_loss=None, metric=None):
        if not self.epochs or self.epochs[-1] != epoch:
            self.epochs.append(epoch)

        if train_loss:
            self.train_losses.append(train_loss)
            self.train_line.set_data(self.epochs, self.train_losses)

        if test_loss:
            self.test_losses.append(test_loss)
            self.test_line.set_data(self.epochs, self.test_losses)

        if metric:
            self.metrics.append(metric)
            self.acc_line.set_data(self.epochs, self.metrics)

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
