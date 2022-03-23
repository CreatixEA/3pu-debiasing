import pickle
import miceforest as mf

import logging
log = logging.getLogger(__name__)

def load_mice_kernel():
    '''
    Load MICE kernel
    '''
    with open('archived/mice_kernel.pkl', 'rb') as f:
        kernel = pickle.load(f)
        log.info("MICE KERNEL LOADED")
        return kernel

def train_mice(df):
    '''
    Train MICE kernel on a dataset and return the kernel
    '''
    kernel = mf.ImputationKernel(data=df, save_all_iterations=True)
    log.debug("Start the Kernel learning")
    kernel.mice(3, verbose=True)
    log.info("MICE kernel trained")

    return kernel

def apply_mice(df, kernel=None):
    ''''
    Apply a trained MICE kernel on a dataset
    '''

    if kernel is None:
        kernel = load_mice_kernel()

    log.debug("Start imputing new data")
    ndf = kernel.impute_new_data(new_data=df, datasets=0, verbose=True).complete_data(0)
    log.info("New data imputed from MICE kernel")

    return ndf

