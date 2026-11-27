import os
os.environ['KERAS_BACKEND'] = 'torch'

from training.network19x19 import get_mucaro_19x19_network




if __name__ == '__main__':
    get_mucaro_19x19_network()