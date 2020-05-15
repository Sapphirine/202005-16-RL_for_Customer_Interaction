
import numpy as np
import pandas as pd
import datetime
from calendar import monthrange
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import argparse
import os
# from preprocessing import preprocess
# from model import data_split,get_lstm,direct_rl,indirect_rl,semidirect_rl
# from keras.models import load_model



pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')




def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='cup98LRN', help='Name of the chosen dataset')

    parser.add_argument('--path-dataset', type=str, default='D:/cup98LRN/cup98LRN.txt', help='Path of dataset')

    parser.add_argument('--method', type=str, default='all', help='Methods to be chosen (all/direct/indirect/semidirect)')

    parser.add_argument('--lstm-epochs', type=int, default=100, help='Number of epochs for lstm (default: 100)')
    parser.add_argument('--lstm-batch-size',  type=int, default=128, help='examples per batch for lstm (default: 128)')

    parser.add_argument('--base-epochs', type=int, default=10, help='Number of epochs for base function (default: 10)')
    parser.add_argument('--base-batch-size',  type=int, default=128, help='examples per batch for base (default: 128)')

    parser.add_argument('--learning-rate', type=float, default=1e-1, help='learning_rate, (default: 0.1)')
    parser.add_argument('--discount-rate', type=float, default=0.9, help='discount_rate, (default: 0.9)')
    parser.add_argument('--iteration', type=int, default=20, help='number of iterations for reinforcement learning')

    parser.add_argument('--episode', type=int, default=95412, help='number of episodes')
    parser.add_argument('--episode-length', type=int, default=23, help='length of episodes')

    return parser.parse_args()



def main():
    args = vars(get_args())
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    #
    # alpha = args['learning-rate']
    # gamma = args['discount-rate']
    # lstm_epochs = args['lstm-epochs']
    # lstm_batch_size = args['lstm-batch-size']
    # base_epochs = args['base-epochs']
    # base_batch_size = args['base-batch-size']
    # episode = args['episode']
    # episode_length = args['episode-length']
    # iteration = args['iteration']
    #
    # raw_df = pd.read_csv(args['path-dataset'])
    #
    # X, y, rewards, s_true = preprocess(raw_df,episode,episode_length)
    #
    # X_train, X_test, y_train, y_test = data_split(X, y)
    # model = get_lstm(X_train, y_train, X_test, y_test,lstm_epochs,lstm_batch_size)
    # model.save('my_model.hdf5')
    #
    #
    #
    # if args['method'] == 'all':
    #     model = load_model('my_model.hdf5')
    #     Q_direct,model_direct = direct_rl(model, X, rewards, alpha, gamma,base_epochs,base_batch_size,episode,episode_length,iteration)
    #     s_direct = np.sum(Q_direct, 1)
    #
    #     Q_indirect,model_indirect = indirect_rl(model, X, rewards,alpha, gamma,base_epochs,base_batch_size,episode,episode_length,iteration)
    #     s_indirect = np.sum(Q_indirect, 1)
    #
    #     Q_semidirect, model_semidirect = semidirect_rl(model, X, rewards,base_epochs,base_batch_size,episode,episode_length,iteration)
    #     s_semidirect = np.sum(Q_semidirect, 1)
    #
    #     s_direct = pd.DataFrame(s_direct)
    #     s_direct.to_csv('/direct_model.csv')
    #     model_direct.save('direct_model.hdf5')
    #
    #     s_indirect = pd.DataFrame(s_indirect)
    #     s_indirect.to_csv('/indirect_model.csv')
    #     model_indirect.save('indirect_model.hdf5')
    #
    #     s_semidirect = pd.DataFrame(s_semidirect)
    #     s_semidirect.to_csv('/semidirect_model.csv')
    #     model_semidirect.save('semidirect_model.hdf5')
    #
    #
    #
    # if args['method'] == 'direct':
    #     model = load_model('my_model.hdf5')
    #     Q_direct,model_direct = direct_rl(model, X, rewards,alpha, gamma,base_epochs,base_batch_size,episode,episode_length,iteration)
    #     s_direct = np.sum(Q_direct, 1)
    #     s_direct = pd.DataFrame(s_direct)
    #     s_direct.to_csv('/direct_model.csv')
    #
    # if args['method'] == 'indirect':
    #     model = load_model('my_model.hdf5')
    #     Q_indirect,model_indirect = indirect_rl(model, X, rewards,alpha, gamma,base_epochs,base_batch_size,episode,episode_length,iteration)
    #     s_indirect = np.sum(Q_indirect, 1)
    #     s_indirect = pd.DataFrame(s_indirect)
    #     s_indirect.to_csv('/indirect_model.csv')
    #
    # if args['method'] == 'semidirect':
    #     Q_semidirect,model_semidirect = semidirect_rl(model, X, rewards,base_epochs,base_batch_size,episode,episode_length,iteration)
    #     s_semidirect = np.sum(Q_semidirect, 1)
    #     s_semidirect = pd.DataFrame(s_semidirect)
    #     s_semidirect.to_csv('/semidirect_model.csv')



if __name__ == '__main__':
     main()
