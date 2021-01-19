import pickle

from ..data import Dataset

def GMM2d(path='./data'):
    with open('{}/GMMs/gmm-2d-syn-set.pkl'.format(path), 'rb') as f:
        raw_dataset = pickle.load(f)

    return Dataset(raw_dataset['x'], raw_dataset['y'])
