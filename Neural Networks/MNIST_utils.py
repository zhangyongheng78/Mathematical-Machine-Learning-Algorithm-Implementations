import numpy as np
import h5py as h5

# Load data
def load_data(filename, dataset):
    with h5.File(filename) as data:
        X = data[dataset + '/data'][:].astype(np.float64)
        N = X.shape[0]
        if dataset != 'kaggle':
            Y = data[dataset + '/labels'][:].astype(np.int32)
            return X, Y
        else:
            return X

# Save kaggle submission
def save_submission(filename, yhats):
    assert np.ndim(yhats) == 1
    id_and_prediction = np.vstack([np.arange(len(yhats)).T, yhats]).T
    np.savetxt(filename, id_and_prediction,
               fmt='%d',
               delimiter=',',
               comments='',
               header='Id,Prediction')
    print("Saved:", filename)
