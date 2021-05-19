import numpy as np
import h5py

# load data
def load_data(filename, group, partition):
    with h5py.File(filename, "r") as hf:
        X = hf[ group + '/' + partition + '/data'][:]
        if partition != 'kaggle':
            Y = hf[ group + '/' + partition + '/labels'][:]
            X = [ X[ Y == c, :] for c in np.unique(Y) ]
        return X

#load experiment
def load_experiment(data_fn, experiment_name):
    Xtrain = load_data(data_fn, experiment_name, "train")
    Xval = load_data(data_fn, experiment_name, "val")
    num_classes = len(Xtrain)
    num_features = Xtrain[0].shape[1]
    for c in range(num_classes):
        assert(num_features == Xtrain[0].shape[1])
    return Xtrain,Xval,num_classes,num_features

class AbstractGenerativeModel(object):
    def __init__(self, CLASSES, NUM_FEATURES):
        self.num_classes = CLASSES
        self.num_features = NUM_FEATURES
        self.params = dict()
    def pack_params(self, output_params, class_idx):
        raise NotImplementedError("Implement this method for both MixtureModel and NaiveBayesModel.")
    def classify(self, X):
        raise NotImplementedError("Implement this method for both MixtureModel and NaiveBayesModel.")
    def fit(self, X, class_idx):
        raise NotImplementedError("Implement this method for both MixtureModel and NaiveBayesModel.")
    def train(self, X):
        for c in range(self.num_classes):
            self.pack_params(self.fit(X[c], c), c)
    def val(self, X, acc=0, N=0):
        for c in range(self.num_classes):
            acc += np.sum((self.classify(X[c]) == c).astype(np.int32))
            N += X[c].shape[0]
        return (acc / N)

# save kaggle submission
def save_submission(filename, yhats):
    assert np.ndim(yhats) == 1
    id_and_prediction = np.vstack([np.arange(len(yhats)).T, yhats]).T
    np.savetxt(filename, id_and_prediction,
            fmt='%d',
            delimiter=',',
            comments='',
            header='Id,Prediction')
    print("Saved:", filename)