import numpy as np
from matplotlib import pyplot as plt


def plot_acc_loss(
    train_acc_series, test_acc_series, train_loss_series, test_loss_series
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training & Validation")

    for line in train_acc_series.items():
        axs[0, 0].plot(line[1], label=line[0])
    axs[0, 0].set_title("Training Accuracy")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Accuracy")
    axs[0, 0].legend()

    for line in train_loss_series.items():
        axs[0, 1].plot(line[1], label=line[0])
    axs[0, 1].set_title("Training Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()

    for line in test_acc_series.items():
        axs[1, 0].plot(line[1], label=line[0])
    axs[1, 0].set_title("Validation Accuracy")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].legend()

    for line in test_loss_series.items():
        axs[1, 1].plot(line[1], label=line[0])
    axs[1, 1].set_title("Validation Loss")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].legend()


def plot_accuracy(train_history, val_history):
    _ = plt.plot(train_history)
    _ = plt.plot(val_history)
    _ = plt.title("Model Accuracy")
    _ = plt.ylabel("Accuracy")
    _ = plt.xlabel("Epoch")
    _ = plt.legend(["train", "val"], loc="upper left")
    _ = plt.show()


def plot_loss(train_history, val_history):
    _ = plt.plot(train_history)
    _ = plt.plot(val_history)
    _ = plt.title("Model Loss")
    _ = plt.ylabel("Loss")
    _ = plt.xlabel("Epoch")
    _ = plt.legend(["train", "val"], loc="upper left")
    _ = plt.show()


def imshow_torch(images):
    np_image = images.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))


def plot_images():
    plt.imshow()
