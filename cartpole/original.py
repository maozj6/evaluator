
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from randomloader import RolloutObservationDataset
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix

class EvaNet(nn.Module):
    def __init__(self):
        super(EvaNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(257, 2)
        self.sft = nn.Softmax()

    def forward(self, x, a):
        input_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(input_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # a=a.view(128,3)
        x = torch.cat((x, a.unsqueeze(1)), 1)
        x = self.fc3(x)
        x = self.sft(x)

        return x

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def main(train_path,test_path,step_save_name,step):


    rnn = EvaNet()
    # Training loop
    test_dataset = RolloutObservationDataset(test_path, leng=step)
    test_dataset.load_next_buffer()

    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)

    rnn_model=torch.load("action0-checkpooint.tar")
    rnn.load_state_dict(rnn_model["state_dict"])
    rnn.to(device)
    y_test = []
    y_pred = []
    for epoch in range(1):
        correct=0
        total_loss = 0
        total_num = 0
        pbar = tqdm(total=len(test_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        rnn.eval()
        for data in test_loader:  # Iterate over your training dataset
            inputs, safes, acts = data
            total_num = total_num + len(inputs)
            inputs = inputs.to(device)
            acts = acts.to(device)
            safes=safes.to(device)
            obs = inputs.float()
            zoutputs = rnn(obs.unsqueeze(1),torch.zeros((batchsize)).to(device))
            _, predicted = torch.max(zoutputs.data, 1)
            y_pred.append(predicted.cpu().detach().numpy())
            y_test.append(safes.cpu().detach().numpy())

            correct += (predicted == safes).sum().item()
            # Forward pass

            # Compute loss

            pbar.update(1)

        # Print the epoch loss

        y_test = np.concatenate(y_test)
        y_pred = np.concatenate(y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        conf=confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf.ravel()
        fpr=fp / (tn + fp)
        print(acc)
        print(f1)
        print(fpr)

        print("end")

if __name__ == '__main__':

    device="cuda"
    print(device)
    batchsize=128
    parser = argparse.ArgumentParser(description='original test')
    parser.add_argument('--test',default="/home/mao/23Summer/code/vision-cartpole-dqn/023dataset/",
                        help='Does not save samples during training if specified')
    parser.add_argument('--save',default="/home/mao/23Summer/code/racing-car/balanced20cnn/actionsave/",
                        help='Does not save samples during training if specified')

    args = parser.parse_args()





    test_path=args.test
    save_path=args.save
    for i in range(0,1):
        main("",test_path,save_path+"action"+str(i),i)




