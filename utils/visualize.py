import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(train_loss,val_loss,save_path):
    plt.figure()
    plt.plot(np.arange(len(train_loss)), train_loss, label="train loss")
    plt.plot(np.arange(len(val_loss)), val_loss, label="val loss")
    plt.legend()
    plt.savefig(save_path,dpi=150)
    plt.close()