import os
import torch.optim as optim
import torch.utils.data as loader
import numpy as np,math
import random
from tqdm import tqdm
from feature import seq_stru_feature
from DCN import *
from torch.utils.data import Dataset
from sklearn import metrics
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader

seed = 0
seed_everything(seed)
random.seed(0)
np.random.seed(0)
# Generate ids for k-flods cross-validation
def Id_k_folds(seqs_num, k_folds, ratio):
    train_ids = []
    test_ids = []
    valid_ids = []
    if k_folds == 1:
        train_num = int(seqs_num * 0.7)
        test_num = seqs_num - train_num
        valid_num = int(train_num * ratio)
        train_num = train_num - valid_num
        index = range(seqs_num)
        train_ids.append(np.asarray(index[:train_num]))
        valid_ids.append(np.asarray(index[train_num:train_num + valid_num]))
        test_ids.append(np.asarray(index[train_num + valid_num:]))
    else:
        each_fold_num = int(math.ceil(seqs_num / k_folds))
        for fold in range(k_folds):
            index = range(seqs_num)
            index_slice = index[fold * each_fold_num : (fold + 1) * each_fold_num]
            index_left = list(set(index) - set(index_slice))
            test_ids.append(np.asarray(index_slice))
            train_num = len(index_left) - int(len(index_left) * ratio)
            train_ids.append(np.asarray(index_left[:train_num]))
            valid_ids.append(np.asarray(index_left[train_num:]))

    return (train_ids, test_ids, valid_ids)



class earlystopping:


    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class SSDataset(Dataset):

    def __init__(self, data_set1, data_set2, labels):
        self.data_set1 = data_set1.astype(np.float32)
        self.data_set2 = data_set2.astype(np.float32)


        self.labels = labels

    def __getitem__(self, item):
        return self.data_set1[item], self.data_set2[item], self.labels[item]

    def __len__(self):
        return self.data_set1.shape[0]


class Constructor:


    def __init__(self, model, stop,lr_rate=1, lr_scheduler = 5,model_name='FNNC1',weight_decay = 0.01):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adadelta(self.model.parameters(),lr=lr_rate,weight_decay = weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=lr_scheduler, verbose=True)
        self.loss_function = nn.BCELoss()
        self.early_stopping = stop
        self.batch_size = 200
        self.epochs = 100
        self.seed = 0

    def learn(self, TrainLoader, ValidateLoader):

        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                xs,xn ,y = data

                output = self.model(xs=xs.to(self.device),xn=xn.to(self.device))
                # print(output.shape)
                loss = self.loss_function(output, y.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())

                loss.backward()
                self.optimizer.step()

            valid_loss = []

            self.model.eval()
            with torch.no_grad():
                for valid_xs,valid_xn, valid_y in ValidateLoader:
                    valid_output = self.model(xs=valid_xs.to(self.device),xn=valid_xn.to(self.device))
                    valid_y = valid_y.float().to(self.device)
                    valid_loss.append(self.loss_function(valid_output, valid_y).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))

                self.scheduler.step(valid_loss_avg)
                print("验证集loss:{}".format(valid_loss_avg))
                self.early_stopping(valid_loss_avg, self.model)
                if self.early_stopping.early_stop:
                    print("此时早停！")
                    break

            # torch.save(self.model.state_dict(), path + '\\' + self.model_name + '.pth')

    def inference(self, TestLoader):
        self.model.load_state_dict(torch.load('checkpoint.pt'))
        predicted_value = []
        ground_label = []
        self.model.eval()
        for xs,xn,y in TestLoader:
            output = self.model(xs.to(self.device),xn.to(self.device))
            """ To scalar"""
            ########################################################################
            predicted_value.append(output.squeeze(dim=0).detach().cpu().numpy().tolist())
            ground_label.append(y.squeeze(dim=0).detach().cpu().numpy().tolist())

        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):

        predicted_value_rounded = np.array(predicted_value).round()
        ground_label_array = np.array(ground_label)

        # 计算各项指标
        sn = metrics.recall_score(y_true=ground_label_array, y_pred=predicted_value_rounded, pos_label=1)
        sp = metrics.recall_score(y_true=ground_label_array, y_pred=predicted_value_rounded, pos_label=0)
        mcc = metrics.matthews_corrcoef(y_true=ground_label_array, y_pred=predicted_value_rounded)
        acc = metrics.accuracy_score(y_true=ground_label_array, y_pred=predicted_value_rounded)
        auroc = metrics.roc_auc_score(y_true=ground_label_array, y_score=np.array(predicted_value))
        f1 = metrics.f1_score(y_true=ground_label_array, y_pred=predicted_value_rounded)
        ap = metrics.average_precision_score(y_true=ground_label_array, y_score=np.array(predicted_value))

        print(sn, sp, mcc, acc, auroc, f1, ap)
        return sn, sp, mcc, acc, auroc, f1, ap

    def run(self, Train_Set, Vaild_Set, Test__Set):

        Train_Loader = stable( loader.DataLoader(dataset=Train_Set, drop_last=True,
                                         batch_size=self.batch_size, shuffle=True, num_workers=0),seed)

        Vaild_Loader = stable(loader.DataLoader(dataset=Vaild_Set, drop_last=True,
                                         batch_size=self.batch_size, shuffle=False, num_workers=0),seed)

        Test_Loader = stable(loader.DataLoader(dataset=Test__Set,
                                        batch_size=1, shuffle=False, num_workers=0),seed)

        self.learn(Train_Loader, Vaild_Loader)
        predicted_value, ground_label = self.inference(Test_Loader)
        print(predicted_value)

        sn, sp, mcc, acc, auroc, f1, ap = self.measure(predicted_value, ground_label)

        return sn, sp, mcc, acc, auroc, f1, ap, predicted_value, ground_label


def act( ):
    K = 10
    ratio_k = 0.1

    path = 'Triticum_aestivum_21.fasta'
    L = 21
    prop = 'c'
    a1=0.9
    a2=0.8
    a3=1


    Y, X1, X2, X3, X4, X5, X6, X7, X8, N1, N2, N3, N4, N5, N6 ,EJS, EBC, ED = seq_stru_feature(path,prop,a1,a2,a3)


    
    # Y = pd.read_csv('labels.csv')
    Y = np.array(Y)
    Y = Y.reshape((Y.shape[0], 1))

    # sequence feature
    X1 = np.array(X1)
    seq_num = X1.shape[0]
    data_bi = X1.reshape((seq_num,L ,-1 ))

    X2 = np.array(X2)
    data_bs = X2.reshape((seq_num,  L,-1))

    X3 = np.array(X3)
    data_aa = X3.reshape((seq_num, L, -1))

    X4 = np.array(X4)
    data_sd = X4.reshape((seq_num, L, -1))

    X5 = np.array(X5)
    data_pc = X5.reshape((seq_num, L, -1))

    X6 = np.array(X6)
    data_egb = X6.reshape((seq_num, L, -1))

    X7 = np.array(X7)
    data_pam = X7.reshape((seq_num, L, -1))

    X8 = np.array(X8)
    data_zs = X8.reshape((seq_num, L, -1))

    # data_Xs_all = [data_bi_CC, data_bs_CC, data_pam_CC, data_zs_CC]

    Xs = np.concatenate((data_bi, data_bs), axis=2)
    # data_bs_CC,data_bi_CC,data_pam_CC,data_zs_CC
    # Xs = data_bi

    Xs_size = Xs.shape[2]

    # network-derived structure feature
    
    # Node importance feature
    N1 = np.array(N1)
    data_PG = N1.reshape((seq_num, L, -1))

    N2 = np.array(N2)
    data_CL = N2.reshape((seq_num, L, -1))

    N3 = np.array(N3)
    data_CC = N3.reshape((seq_num, L,-1))

    N4 = np.array(N4)
    data_BC = N4.reshape((seq_num, L, -1))

    N5 = np.array(N5)
    data_EC = N5.reshape((seq_num, L, -1))

    N6 = np.array(N6)
    data_DC = N6.reshape((seq_num, L, -1))

    Xnn = np.concatenate((data_PG, data_CL, data_CC, data_BC, data_EC, data_DC), axis=2)
    # edge importance feature
    data_ejs =np.array(EJS)
    data_ebc = np.array(EBC)
    data_ed = np.array(ED)
    # Xn = np.concatenate((data_ebc, data_ed), axis=2)
    # data_ebc, data_ed, data_ejs,Xnn
    Xn = data_ebc
    Xn_size = Xn.shape[2]





    # 10-folds cross-validation
    indices = np.arange(seq_num)
    np.random.seed(0)
    np.random.shuffle(indices)

    Xs_data_train = Xs[indices]
    Xn_data_train = Xn[indices]

    intensity_train = Y[indices]
    train_ids, test_ids, valid_ids = Id_k_folds(seq_num, k_folds=K, ratio=ratio_k)

    Sn = []
    Sp = []
    Acc = []
    Mcc = []
    auROC = []
    F1 = []
    Ap = []
    pre_value = []
    true_label = []
    start_time = time.time()

    for fold in range(K):


        xs_train = Xs_data_train[train_ids[fold]]
        xn_train = Xn_data_train[train_ids[fold]]


        y_train = intensity_train[train_ids[fold]]

        xs_valid = Xs_data_train[valid_ids[fold]]
        xn_valid = Xn_data_train[valid_ids[fold]]

        y_valid = intensity_train[valid_ids[fold]]

        xs_test = Xs_data_train[test_ids[fold]]
        xn_test = Xn_data_train[test_ids[fold]]

        y_test = intensity_train[test_ids[fold]]

        Train_Set = SSDataset(data_set1=xs_train, data_set2=xn_train, labels=y_train)
        Vaild_Set = SSDataset(data_set1=xs_valid, data_set2=xn_valid,  labels=y_valid)
        Test__Set = SSDataset(data_set1=xs_test, data_set2=xn_test, labels=y_test)

        early_stopping = earlystopping(patience=10, verbose=True)


        Train = Constructor(model=DenseBlockModel_SE(in_1=Xs_size, in_2=Xn_size, denseblocks=4, layers=2, filters=192,growth_rate=64, dropout_rate=0,ratio=0.3,fun ='prelu'),
                            stop=early_stopping,lr_rate=1,lr_scheduler=4,weight_decay=0.01)



        print("\n_______________fold", fold, "_____________\n")
        # Calculate the running time
        sn, sp, mcc, acc, auroc, f1, ap, predicted_value, ground_label = Train.run(Train_Set, Vaild_Set, Test__Set)
        Sn.append(sn)
        Sp.append(sp)
        Mcc.append(mcc)
        Acc.append(acc)
        auROC.append(auroc)
        F1.append(f1)
        Ap.append(ap)

        pre_value += np.array(predicted_value).flatten().tolist()
        true_label += np.array(ground_label).flatten().tolist()

    end_time = time.time()
    total_time = end_time - start_time
    print("total_time:", total_time / K)
    print(pre_value)
    print(true_label)
    print(" Sn: {}  Sp: {}  Mcc: {}  Acc: {}  auROC: {}  F1: {}  Ap: {}".format(
        np.mean(Sn), np.mean(Sp), np.mean(Mcc), np.mean(Acc), np.mean(auROC), np.mean(F1), np.mean(Ap)))

    return ([np.mean(Sn), np.mean(Sp), np.mean(Mcc), np.mean(Acc), np.mean(auROC), np.mean(F1), np.mean(Ap)])



if __name__ == '__main__':
    act()
