import matplotlib.pyplot as plt


class LivePlot:
    def __init__(self):
        plt.ion()  # interactive mode
        self.fig, self.ax = plt.subplots()
        self.epochs = []
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []

        (self.train_line,) = self.ax.plot([], [], label="Train Loss")
        (self.test_line,) = self.ax.plot([], [], label="Test Loss")
        (self.acc_line,) = self.ax.plot([], [], label="Accuracy")

        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss / Accuracy")
        self.ax.legend()
        self.ax.grid(True)

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

        # Rescale axes dynamically
        self.ax.relim()
        self.ax.autoscale_view()

        # Draw updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # short pause so GUI updates

    def show(self):
        plt.ioff()
        plt.show()
