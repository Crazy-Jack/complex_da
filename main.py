import numpy as np
import argparse
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from dataset import TimeSeriesDataset
from complex_transformer import ComplexTransformer
from fnn import FNN

parser = argparse.ArgumentParser(description='Time series adaptation')
parser.add_argument("--data-path", type=str, default="/projects/rsalakhugroup/complex/domain_adaptation", help="dataset path")
parser.add_argument("--task", type=str, default="3Av2", help='3Av2 or 3E')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--seed', type=int, help='manual seed')

args = parser.parse_args()

device = torch.device("cuda:0")

# seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
print("seed", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True


def main():
    training_set = TimeSeriesDataset(root_dir=args.data_path, file_name="processed_file_{}.pkl".format(args.task), train=True)
    test_set = TimeSeriesDataset(root_dir=args.data_path, file_name="processed_file_{}.pkl".format(args.task), train=False)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    
    seq_len = 10 
    feature_dim = 160

    encoder = ComplexTransformer(layers=1, 
                               time_step=seq_len, 
                               input_dim=feature_dim, 
                               hidden_size=512, 
                               output_dim=512, 
                               num_heads=8,
                               out_dropout=0.5)
    encoder.to(device)
    model = FNN(d_in=feature_dim * 2 * seq_len, d_h=500, d_out=50, dp=0.5).to(device)
    params = list(encoder.parameters()) + list(model.parameters())

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    # Encoding by complex transformer
    best_acc_train = best_acc_test = 0
    for epoch in range(args.epochs):
        """Training"""
        correct_train = 0
        total_bs_train = 0 # total batch size
        train_loss = 0
        for batch_id, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # take the real and imaginary part out
            real = x[:,:,0].reshape(x.shape[0], seq_len, feature_dim).float().to(device)
            imag = x[:,:,1].reshape(x.shape[0], seq_len, feature_dim).float().to(device)
            real, imag = encoder(real, imag)
            pred = model(torch.cat((real, imag), -1).reshape(x.shape[0], -1)) 
            loss = criterion(pred, y.argmax(-1))
            #print(pred.argmax(-1), y.argmax(-1))
            correct_train += (pred.argmax(-1) == y.argmax(-1)).sum().item()
            total_bs_train += y.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
        train_acc = float(correct_train) / total_bs_train
        train_loss = train_loss / total_bs_train
        best_acc_train = max(best_acc_train, train_acc)

        """Testing"""
        correct_test = 0
        total_bs_test = 0
        test_loss = 0
        for batch_id, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            model.eval()

            with torch.no_grad():
                real = x[:,:,0].reshape(x.shape[0], seq_len, feature_dim).float().to(device)
                imag = x[:,:,1].reshape(x.shape[0], seq_len, feature_dim).float().to(device)
                real, imag = encoder(real, imag)
                pred = model(torch.cat((real, imag), -1).reshape(x.shape[0], -1)) 
                loss = criterion(pred, y.argmax(-1))
                correct_test += (pred.argmax(-1) == y.argmax(-1)).sum().item()
                total_bs_test += y.shape[0]
                test_loss += loss.item() * x.shape[0]
        test_acc = float(correct_test) / total_bs_test
        test_loss = test_loss / total_bs_test

        best_acc_test = max(best_acc_test, test_acc)


        #print("train loss", train_loss, "test_loss", test_loss)
        #print("train acc", train_acc, "test_acc", test_acc)

    print(best_acc_train, best_acc_test)
        #scheduler.step(test_loss)
if __name__ == "__main__":
    main()

