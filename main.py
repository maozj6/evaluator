
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from randomloader import RolloutObservationDataset
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix


def main(train_path,test_path,step_save_name,step):


    # Training loop
    test_dataset = RolloutObservationDataset(test_path, leng=step)
    test_dataset.load_next_buffer()
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    #y_test is for labels
    y_test = []
    # y_pred is for predictions(safety)
    y_pred = []
    for epoch in range(1):

        pbar = tqdm(total=len(test_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        for data in test_loader:  # Iterate over your training dataset
            imgs, safes, acts = data



            y_pred.append()
            y_test.append(safes.cpu().detach().numpy())


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
    parser.add_argument('--test',default="013dataset/",
                        help='the path of test dataset')

    args = parser.parse_args()
    test_path=args.test
    save_path=args.save
    for i in range(0,1):
        main("",test_path,"",i)




