# This is a sample Python script.
from vae import VAE

from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix

import torch.nn.functional as f
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from randomloader import RolloutObservationDataset
import argparse
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class EvaNet(nn.Module):
    def __init__(self):
        super(EvaNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256,2)
        self.soft=nn.Softmax(dim=1)

    def forward(self, x):
        input_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(input_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x=self.soft(x)
        return x


if __name__ == '__main__':

    device="cuda"
    print(device)
    batchsize=128
    parser = argparse.ArgumentParser(description='VAE Trainer')

    parser.add_argument('--test',default="/home/mao/23Spring/cars/half_pre/013dataset/",
                        help='Does not save samples during training if specified')

    args = parser.parse_args()

    vae = VAE(1, 32).to(device)
    best = torch.load("safe_vae_best.tar")
    vae.load_state_dict(best["state_dict"])


    vae.eval()

    decoder=vae.decoder

    test_path=args.test
    rnn=EvaNet()

    rnn_model=torch.load("lightaug0-best.tar")
    rnn.load_state_dict(rnn_model["state_dict"])
    rnn.to(device)
    test_dataset = RolloutObservationDataset(test_path, leng=0)
    test_dataset.load_next_buffer()
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    rnn = rnn.to(device)
    train_loss = []
    test_loss = []

    y_test = []
    y_pred = []
    for epoch in range(1):

        total_loss = 0
        total_num = 0
        pbar = tqdm(total=len(test_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        rnn.train()
        for data in test_loader:  # Iterate over your training dataset
            inputs ,safes,acts = data
            total_num = total_num + len(inputs)
            inputs = inputs.to(device)
            obs = inputs.float()
            obs, obs2 = [
                f.upsample(x.view(-1, 1, 64, 64), size=64,
                           mode='bilinear', align_corners=True)
                for x in (obs, obs)]

            obs_z,obsz2 = [
                vae(x)[1] for x in (obs, obs2)]
            # zs=vae(inputs)
            zoutputs = rnn(obs_z)
            # Forward pass

            pbar.update(1)
            _, predicted = torch.max(zoutputs.data, 1)
            y_pred.append(predicted.cpu().detach().numpy())
            y_test.append(safes.cpu().detach().numpy())


        # Print the epoch loss
        y_test = np.concatenate(y_test)
        y_pred = np.concatenate(y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        conf = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf.ravel()
        fpr = fp / (tn + fp)
        print(acc)
        print(f1)
        print(fpr)

        print("end")
