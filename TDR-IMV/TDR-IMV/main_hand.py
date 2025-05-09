import os
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import Scene, PIE, CUB, handwritten, ROSMAP, BRAC, Generate

from model import TDR_IMV

from util import mv_dataset, read_mymat, build_ad_dataset, process_data, get_validation_set, mv_tabular_collate, \
    get_validation_set_Sn, partial_mv_dataset, partial_mv_tabular_collate, Uncertainty_filter
import warnings
from EarlyStopping_hand import EarlyStopping
from collections import Counter
from select_k_neighbors import get_samples
from loss_function import get_loss
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training [default: 200]')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train [default: 3]')
    parser.add_argument('--preepoch', type=int, default=10, metavar='N',
                        help='number of epochs to train [default: 10]')
    parser.add_argument('--annealing-epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate [default: 1e-2 or 3e-3 1e-3]')
    parser.add_argument('--patience', type=int, default=3, metavar='LR',
                        help='parameter of Earlystopping [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: False]')
    parser.add_argument('--missing-rate', type=float, default=0.1, metavar='LR',
                        help='missingrate [default: 0.1,0.3,0.5]')
    parser.add_argument('--n-sample', type=int, default=30, metavar='LR',
                        help='times of sampling [default: 30]')
    parser.add_argument('--k', type=int, default=5, metavar='LR',
                        help='number of neighbors [default: 5]')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    args = parser.parse_args()

    dataset = BRAC()
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    dataset_name = dataset.data_name+".mat"
    delta = 1
    gamma = 1
    beta = 1.5
    test_time = 1

    missing_rate = args.missing_rate
    X, Y, Sn = read_mymat('./data/', dataset_name, ['X', 'Y'], missing_rate)
    partition = build_ad_dataset(Y, p=0.8, seed=999)
    X = process_data(X, num_views)
    X_train, Y_train, X_test, Y_test, Sn_train = get_samples(x=X, y=Y, sn=Sn,
                                                             train_index=partition['train'],
                                                             test_index=partition['test'],
                                                             n_sample=args.n_sample,
                                                             k=args.k)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=partial_mv_dataset(X_train, Sn_train, Y_train),
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=1,
                              collate_fn=partial_mv_tabular_collate)

    def pretrain(epoch):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, sn, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].float().cuda())
            target = Variable(target.long().cuda())
            sn = Variable(sn.long().cuda())
            # refresh the optimizer
            optimizer.zero_grad()
            evidences, evidence_a, evidence_con, evidence_div = model(data, target, epoch, beta)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            loss = get_loss(evidences, evidence_a, evidence_con, evidence_div, target, epoch, num_classes,
                            args.annealing_epochs,
                            gamma, delta, device)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
        print('Pretrain Epoch: {} \tLoss: {:.6f}'.format(epoch, loss_meter.avg))


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, sn, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].float().cuda())
            target = Variable(target.long().cuda())
            sn = Variable(sn.long().cuda())
            # refresh the optimizer
            optimizer.zero_grad()

            evidences, evidence_a, evidence_con, evidence_div = model(data, target, epoch, beta)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            loss = get_loss(evidences, evidence_a, evidence_con, evidence_div, target, epoch, num_classes,
                            args.annealing_epochs,
                            gamma, delta, device)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
        return loss_meter.avg

    def test(epoch, dataloader):
        model.eval()
        num_correct, num_sample, = 0, 0,
        loss_meter_2 = AverageMeter()
        for batch_idx, (data, target) in enumerate(dataloader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].float().cuda())
            target = Variable(target.long().cuda())
            num_sample += target.size(0)
            with torch.no_grad():
                evidence, evidences, evidence_con, evidence_div = model(data, target, epoch, beta)
                _, Y_pre = torch.max(evidences, dim=1)
                num_correct += (Y_pre == target).sum().item()
                loss = get_loss(evidence, evidences, evidence_con, evidence_div, target, epoch, num_classes,
                                args.annealing_epochs,
                                gamma, delta, device)
                loss_meter_2.update(loss.item())
        acc = num_correct / num_sample
        return acc

    results = []
    losss = []
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    model = TDR_IMV(num_views, dims, num_classes, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.cuda:
        model.cuda()
    for epo in range(1, args.preepoch):
        pretrain(epo)
    print("Sampled points filtering...")
    X_filter_train_, Y_filter_train_ = Uncertainty_filter(X_train, Y_train, model, num_classes, args.preepoch, beta, args)
    X_train, Y_train = X_filter_train_, Y_filter_train_
    train_loader = DataLoader(dataset=partial_mv_dataset(X_train, Sn_train, Y_train),
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=1,
                              collate_fn=partial_mv_tabular_collate)
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        losss.append(loss)
        early_stopping(loss * (-1), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('Train Epoch: {} Loss: {:.6f}'.format(
            epoch, loss))
    X_filter_test, Y_filter_test = Uncertainty_filter(X_test, Y_test, model, num_classes, epoch, beta, args)
    test_loader = DataLoader(dataset=mv_dataset(X_filter_test, Y_filter_test), batch_size=10,
                             shuffle=False, num_workers=1,
                             collate_fn=mv_tabular_collate)
    acc = test(1, test_loader)
    results.append(acc)
    print('===========================================')
    print(results)

    with open(f"./{dataset_name}.txt", "a") as f:
        text = "\tmissing_rate:" + str(missing_rate) + "\taccuracy:" + str(results)
        f.write(text)
    f.close()