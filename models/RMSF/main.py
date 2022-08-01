"""
Description: Machine Learning
Author - Mayank Agrawal

USAGE:
python main.py --train train.csv test.csv --model-path model.tar --new
python main.py --predict test.csv --model-path model.tar

"""



################################### LIBRARIES #####################################
import os, time
import copy
import argparse
import numpy as np
import torch
import pandas as pd
import sklearn.model_selection
import tqdm
###################################################################################


################################### ARGPARSE #####################################
msg = "Main script for ML."
parser = argparse.ArgumentParser(description = msg)
parser.add_argument("--collect-data", nargs=2, help = "Prepare Train and Test csv data. Accepts <filename> <propname> in this order")
parser.add_argument("--train-test-split", help = "Prepare Train and Test data. Creates csv files.", nargs=2)
parser.add_argument("--train", help = "Train using the given data file", nargs=2)
parser.add_argument("--model-path", help = "Model path to use. ex. model.tar")
parser.add_argument("--new", help = "initiate a new model", action='store_true')
parser.add_argument("--confusion", help = "Calculate confusion table for the given data (in csv format).")
parser.add_argument("--plots", help = "Plot using the evaluation tar file generated while training")
parser.add_argument("--predict", help = "Predict using the given data (in csv format).")
parser.add_argument("--create-multiple-sets", action='store_true')
args = parser.parse_args()
###################################################################################


################################### DATAPREP ######################################

def load_props(filename, prop_name):
    dataframe = pd.read_csv(filename, sep=',', header=0)
    prop = dataframe[prop_name].values
    PA = dataframe['PA'].values
    seqs = np.array([s.replace('C16','') for s in PA])
    filtr = prop == prop
    prop = prop[filtr]
    seqs = seqs[filtr]
    return seqs, prop


def collect_data(args):
    """ Collect and prepare train and test data """
    filename = args[0]
    prop_name = args[1]
    p = load_props(filename, prop_name)
    all_seqs, prop = p[0], p[1]
    prop = np.round(prop, 4)
    args = np.argsort(prop)
    
    num_true = int(len(args)/3)
    args_true = args[-num_true:]
    args_false = args[:num_true]
    
    seqs = np.append(all_seqs[args_false], all_seqs[args_true])
    y = np.append(np.zeros(len(args_false)), np.ones(len(args_true)))
    seq_train, seq_test, y_train, y_test = sklearn.model_selection.train_test_split(seqs, y, test_size=0.2, random_state=42)
    
    data = np.array([seq_train, y_train]).T
    pd.DataFrame(data).to_csv('train.csv', sep=',', header=None, index=False)
    data = np.array([seq_test, y_test]).T
    pd.DataFrame(data).to_csv('test.csv', sep=',', header=None,  index=False)



def train_test_split(filename, test_size=0.2):
    """ split the sequences into training and test set and save csv files """

    data = pd.read_csv(filename, sep=',', header=None).values
    all_seqs = data[:,0]
    y = data[:,1]
    
    seq_train, seq_test, y_train, y_test = sklearn.model_selection.train_test_split(all_seqs, y, test_size=test_size, random_state=42)

    data = np.array([seq_train, y_train]).T
    pd.DataFrame(data).to_csv('train.csv', sep=',', header=None, index=False)
    
    data = np.array([seq_test, y_test]).T
    pd.DataFrame(data).to_csv('test.csv', sep=',', header=None,  index=False)




###################################################################################

################################## ML ARCHITECTURE ################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]



def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.LSTM):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)



class LSTM(torch.nn.Module):
    """ ML model architecture
    """

    def __init__(self, input_size, output_size, hidden_size=50, dropout_lstm=0.8, dropout_fc=0.8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.lstm = torch.nn.LSTM(input_size, self.hidden_size, self.num_layers, dropout=dropout_lstm, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_fc),
            torch.nn.Linear(self.hidden_size, output_size),
            torch.nn.Sigmoid()
            )
            
        self.num_params = 0
        self.num_params += sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        self.num_params += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)

    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers,len(x),self.hidden_size)
        c0 = torch.zeros(self.num_layers,len(x),self.hidden_size)
        
        out, (h, c) = self.lstm(x, (h0, c0))
        z = self.fc(out[:,-1,:])
        return z

###################################################################################

################################## MODEL CLASS #######################################

class Model():
    """ Main class """
    
    def __init__(self, model_path='model.tar'):
        # parameters
        self.BATCH_SIZE    = 64
        self.N_EPOCHS      = 200
        self.LR            = 1e-3
        
        self.INPUT_DIM     = 20
        self.OUTPUT_DIM    = 1
        
        self.model_path    = model_path
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_class = LSTM
        
        
        self.model = None
        self.last_loss = None
        self.last_precision = None
        self.checkpoint = {}


    def one_hot_encode(self, filename):
        """ one-hot encode the letter sequence """
        
        dataframe = pd.read_csv(filename, sep=',', header=None)
        seqs = dataframe.values[:,0]
        
        y = dataframe.values[:,1]
        y = y.reshape(-1,1).astype(float)

        # Residue Dictionary
        residues = ['G', 'A', 'V', 'S', 'T', 'L', 'I', 'M', 'P', 'F', 'Y', 'W', 'N', 'Q', 'H', 'K', 'R', 'E', 'D', 'C']
        res_dict = {}
        for i,r in enumerate(residues):
            res_dict[r] = i
        max_peplen = 10

        X = np.zeros((len(seqs), max_peplen, len(res_dict)))
        for i,seq in enumerate(seqs):
            for j,res in enumerate(seq[:max_peplen]):
                X[i,j,res_dict[res]] = 1
        
        X = X.astype(float)
        
        print(f"Sample Distribution: {len(X)} ({np.sum(y)/len(y)})")

        return X, y, seqs


    def new_model(self):
        self.model = self.model_class(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.last_loss = 1000
        self.last_precision = 0
        self.checkpoint = {}

        
    def load_model(self):
        self.checkpoint = torch.load(self.model_path)
        self.model = self.model_class(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.model.load_state_dict(self.checkpoint['last_model_state_dict'])
        try:
            self.last_precision = checkpoint['last_precision']
        except:
            self.last_precision = 0
        try:
            self.last_loss = checkpoint['last_loss']
        except:
            self.last_loss = 1000


    def init_loss_function(self):
        self.loss_function = torch.nn.BCELoss(reduction='mean')


    def init_optimizer_function(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)


    def train_step(self, train_iterator):
        # set the train mode
        self.model.train()

        # loss of the epoch
        train_loss = []

        for i, (X, y) in enumerate(train_iterator):
            X = X.float()
            y = y.float()

            X = X.to(self.device)

            # update the gradients to zero
            self.optimizer.zero_grad()

            # forward pass
            z = self.model(X)
            
            # loss
            loss = self.loss_function(z, y)
            
            # backward pass
            loss.backward()
            
            train_loss += [loss.item()]
            
            # update the weights
            self.optimizer.step()

        return np.mean(train_loss)



    def test_step(self, test_iterator):
        # set the evaluation mode
        self.model.eval()

        # test loss for the data
        test_loss = []

        # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
        with torch.no_grad():
            for i, (X, y) in enumerate(test_iterator):
                X = X.float()
                y = y.float()
                
                X = X.to(self.device)

                # forward pass
                z = self.model(X)

                # loss
                loss = self.loss_function(z, y)

                # total loss
                test_loss += [loss.item()]

        return np.mean(test_loss)


    def accuracy(self, X, y):
        """ Accuracy """
    
        self.model.eval()
        z = self.model( torch.tensor(X).float() ).detach().numpy()
        
        correct = y == np.round(z)
        denominator = len(correct)

        if len(correct)==0:
            accu = 0.0
        else:
            accu = 100 * np.sum(correct) / denominator
        
        return accu


    def confusion(self, X, y):
        """ confusion matrix calculation """
        
        self.model.eval()
        z = self.model( torch.tensor(X).float() ).detach().numpy()
        z = np.round(z)

        P = sum(y)
        N = len(y) - P
        TP = sum(y*z)
        FP = sum((1-y)*z)
        TN = sum((1-y)*(1-z))
        FN = sum(y*(1-z))

        return P, N, TP, FP, TN, FN


    def train(self, train_datafile, test_datafile):
        # train the model

        f = self.one_hot_encode(train_datafile)
        X_train, y_train = f[0], f[1]
        f = self.one_hot_encode(test_datafile)
        X_test, y_test = f[0], f[1]

        # optimizer
        self.init_optimizer_function()

        # init loss function
        self.init_loss_function()

        # Create iterators for train and test data
        train_dataset = Dataset(X_train, y_train)
        test_dataset = Dataset(X_test, y_test)
        train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.BATCH_SIZE)


        print(f'Num trainable params: {self.model.num_params}, Num train samples: {len(X_train)}')
        accuracy_train = self.accuracy(X_train, y_train)
        accuracy_test = self.accuracy(X_test, y_test)
        print(f'Accuracy - Train: {accuracy_train:.1f}, Test: {accuracy_test:.1f}')
        evaluation = dict(train_loss=[], validation_loss=[], test_loss=[], 
            accuracy_train=[], accuracy_validation=[], accuracy_test=[])
        for e in tqdm.tqdm(range(self.N_EPOCHS)):
            train_loss = self.train_step(train_iterator)
            test_loss  = self.test_step(test_iterator)

            accuracy_train = self.accuracy(X_train, y_train)
            accuracy_test  = self.accuracy(X_test, y_test)
            P, N, TP, FP, TN, FN = self.confusion(X_test, y_test)
            precision = 100 * TP[0] / ( TP[0] + FP[0] )
            print(f'Epoch {e:3d} Loss (Accu): Train {train_loss:.3f}({accuracy_train:.1f}), Test {test_loss:.2f}({accuracy_test:.1f}), Precision {precision:.1f}')

            evaluation['train_loss'] += [train_loss]
            evaluation['test_loss'] += [test_loss]
            evaluation['accuracy_train'] += [accuracy_train]
            evaluation['accuracy_test'] += [accuracy_test]

            # save
            self.checkpoint['last_model_state_dict'] = self.model.state_dict()
            if test_loss < self.last_loss:
                self.checkpoint['best_model_state_dict'] = copy.deepcopy(self.model.state_dict())
                self.checkpoint['loss'] = test_loss
                self.last_loss = test_loss
            
            if precision > self.last_precision:
                # self.checkpoint['best_model_state_dict'] = copy.deepcopy(self.model.state_dict())
                self.checkpoint['precision'] = precision
                self.last_precision = precision

            torch.save(self.checkpoint, self.model_path)
            torch.save(evaluation, 'evaluation.tar')


    def predict(self, X):
        self.model.eval()
        z = self.model( torch.tensor(X).float() ).detach().numpy()
        return z



###################################################################################

###################################### PLOTS ######################################

def plots(filename):
    """ make plots 
    Takes the evaluation.tar file
    """
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    evaluation = torch.load(filename)
    train_loss = evaluation['train_loss']
    test_loss = evaluation['test_loss']
    validation_loss = evaluation['validation_loss']
    accuracy_train = evaluation['accuracy_train']
    accuracy_test = evaluation['accuracy_test']
    accuracy_validation = evaluation['accuracy_validation']
    
    filename_ = 'loss.png'
    fig = plt.figure(figsize=(3.5/1.2,3/1.2))
    x = range(len(train_loss))
    y = train_loss
    plt.plot(x, y, label='Train')
    y = validation_loss
    plt.plot(x, y, label='Valid')
    y = test_loss
    plt.plot(x, y, label='Test')
    # plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax = plt.gca()
    ax.set_xticklabels(range(0,101,25))
    ax.set_xticklabels(range(0,101,25))
    # ax.set_xticks(ticks=x[::25], minor=True)
    plt.xlim(0,100)
    # plt.legend(frameon=False, loc='upper right', fontsize='x-small')
    plt.legend(frameon=False, fontsize='small')
    plt.subplots_adjust(bottom=0.18, left=0.2, top=0.85)
    plt.savefig(filename_, dpi=400)
    os.system(f'open {filename_}')
    plt.close()

    filename_ = 'accuracy.png'
    fig = plt.figure(figsize=(3.5/1.2,3/1.2))
    x = range(len(accuracy_train))
    y = accuracy_train
    plt.plot(x, y, label='Train')
    y = accuracy_validation
    plt.plot(x, y, label='Valid')
    y = accuracy_test
    plt.plot(x, y, label='Test')
    # plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    ax = plt.gca()
    ax.set_xticklabels(range(0,101,25))
    plt.xlim(0,100)
    # ax.set_xticks(ticks=x[::5], minor=True)
    plt.legend(frameon=False, fontsize='small')
    plt.subplots_adjust(bottom=0.18, left=0.2, top=0.85)
    plt.savefig(filename_, dpi=400)
    os.system(f'open {filename_}')
    plt.close()

###################################################################################


###################################### MAIN #######################################

def main():
    """ do as the flags say """
    
    
    if args.collect_data:
        collect_data(args.collect_data)
    

    if args.create_multiple_sets:
        create_multiple_sets()


    if type(args.train_test_split) != type(None):
        train_test_split(filename=args.train_test_split[0], test_size=float(args.train_test_split[1]))


    if type(args.model_path) != type(None):
        model = Model(model_path=args.model_path)
        if args.new:
            model.new_model()
        else:
            model.load_model()


    if type(args.train) != type(None):        
        model.train(args.train[0], args.train[1])


    if type(args.confusion) != type(None):
        datafile_ = args.confusion
        model.model.load_state_dict(model.checkpoint['best_model_state_dict'])
        f = model.one_hot_encode(datafile_)
        X, y = f[0], f[1]        
        P, N, TP, FP, TN, FN = model.confusion(X, y)
        print(f'Confusion Table: P - {P} | N - {N}')
        print(f'|   TP {TP}   |   FN {FN}   |')
        print(f'|   FP {FP}   |   TN {TN}   |')
        precision = 100 * TP[0] / ( TP[0] + FP[0] )
        accuracy = 100 * (TP[0]+TN[0]) / (P[0]+N[0])
        print(f'Precision {precision} Accuracy {accuracy}')


    if type(args.plots) != type(None):
        datafile_ = args.plots
        plots(datafile_)


    if type(args.predict) != type(None):
        datafile_ = args.predict
        model.model.load_state_dict(model.checkpoint['best_model_state_dict'])
        f = model.one_hot_encode(datafile_)
        X, y, seqs = f[0], f[1], f[2]
        accu = model.accuracy(X, y)
        z = np.round(model.predict(X),1)
        model.init_loss_function()
        loss = model.loss_function(torch.tensor(z).float(), torch.tensor(y).float()) / len(y)
        print(f'Prediction Accuracy (%): {accu} | Loss: {loss}')
        data = np.concatenate([seqs.reshape(-1,1), y, z], axis=1)
        file_predict = datafile_.replace('.csv','')+'_predict.csv'
        pd.DataFrame(data).to_csv(file_predict, sep=',', header=None, index=False)



if __name__ == '__main__':
    main()

###########################################################################




