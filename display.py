import numpy as np # for math & importing data
from matplotlib import pyplot as plt # for plotting/saving data
from tqdm import tqdm # for progress bar

import pathlib # for finding current file path
from time import time # for timing the training process

if __name__ == '__main__':

    # data = []
    # with open('loss-stylegan2.npy', "rb") as f:
    #     try:
    #         while True:
    #             disc_loss = np.load(f)
    #             data.append(disc_loss)
    #             gen_loss = np.load(f)
    #             data.append(gen_loss)
    #             divergence = np.load(f)
    #             data.append(divergence)
    #     except:
    #         print('done loading')
    # data = np.array(data)
    # d1 = data[::3] # disc loss
    # d2 = data[1::3] # generator loss
    # d3 = data[2::3] # divergence




    d1 = np.ones(1)
    d2 = np.ones(1)
    with open('test_losses.npy', "rb") as f:
        d1 = np.load(f).sum(axis=1).sum(axis=1).sum(axis=1)
    with open('training_losses.npy', "rb") as f:
        d2 = np.load(f).sum(axis=1).sum(axis=1).sum(axis=1).reshape((-1,d1.shape[0])).sum(axis=0)
    x = np.arange(1, d1.shape[0]+1)




    plt.plot(x,d1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Our Network Test Loss Per Epoch")
    plt.draw() # must do draw() then show() other wise it breaks
    plt.savefig(str(pathlib.Path().absolute()) + '/lstm_test_losses.png')
    plt.show()

    x = np.arange(1, d2.shape[0]+1)
    plt.plot(x,d2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Our Network Training Loss Per Epoch")
    plt.draw() # must do draw() then show() other wise it breaks
    plt.savefig(str(pathlib.Path().absolute()) + '/lstm_training_losses.png')
    plt.show()

    # plt.plot(d3)
    # plt.xlabel("Pass")
    # plt.ylabel("Mistakes")
    # plt.title("Mistakes Per Pass")
    # plt.show()
    # print(d1.shape)
    # print(d2.shape)
    exit()
