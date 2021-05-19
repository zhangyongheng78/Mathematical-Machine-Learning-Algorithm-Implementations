import numpy as np

# For gradient check
def relative_error(x, y, h):
    h = h or 1e-12
    if type(x) is np.ndarray and type(y) is np.ndarray:
        top = np.abs(x - y)
        bottom = np.maximum(np.abs(x) + np.abs(y), h)
        return np.amax(top/bottom)
    else:
        return abs(x - y) / max(abs(x) + abs(y), h)


def numeric_gradient(f, x, df, eps):
    df = df or 1.0
    eps = eps or 1e-8
    n = x.size
    x_flat = x.reshape(n)
    dx_num = np.zeros(x.shape)
    dx_num_flat = dx_num.reshape(n)
    for i in range(n):
        orig = x_flat[i]
    
        x_flat[i] = orig + eps
        pos = f(x)
        if type(df) is np.ndarray:
            pos = pos.copy()
    
        x_flat[i] = orig - eps
        neg = f(x)
        if type(df) is np.ndarray:
            neg = neg.copy()

        d = (pos - neg) * df / (2 * eps)
        
        dx_num_flat[i] = d
        x_flat[i] = orig
    return dx_num

#Criterion for testing the modules
class TestCriterion(object):
    def __init__(self):
        return
        
    def forward(self, _input, _target):
        return np.mean(np.sum(np.abs(_input), 1))
    
    def backward(self, _input, _target):
        self._gradInput = np.sign(_input) / len(_input)
        return self._gradInput

def sgd(x, dx, lr, weight_decay = 0):
    if type(x) is list:
        assert len(x) == len(dx), 'Should be the same'
        for _x, _dx in zip(x, dx):
            sgd(_x, _dx, lr)
    else:
        x -= lr * (dx + 2 * weight_decay * x)  

def sgdm(x, dx, lr, alpha = 0.8 , state = None, weight_decay = 0):
    # sgd with momentum, standard update
    if not state:
        if type(x) is list:
            state = [None] * len(x)
        else:
            state = {}
            state['v'] = np.zeros(x.shape)
    if type(x) is list:
        for _x, _dx, _state in zip(x, dx, state):
            sgdm(_x, _dx, lr, alpha, _state)
    else:
        state['v'] *= alpha
        state['v'] += lr * (dx + 2 * weight_decay * x)  
        x -= state['v']

def sgdmom(x, dx, lr, alpha = 0, state = None, weight_decay = 0):
    # sgd momentum, uses nesterov update (reference: http://cs231n.github.io/neural-networks-3/#sgd)
    if not state:
        if type(x) is list:
            state = [None] * len(x)
        else:
            state = {}
            state['m'] = np.zeros(x.shape)
            state['tmp'] = np.zeros(x.shape)
    if type(x) is list:
        for _x, _dx, _state in zip(x, dx, state):
            sgdmom(_x, _dx, lr, alpha, _state)
    else:
        state['tmp'] = state['m'].copy()
        state['m'] *= alpha
        state['m'] -= lr * (dx + 2 * weight_decay * x)  
        
        x -= alpha * state['tmp']
        x += (1 + alpha) * state['m']
        
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
        