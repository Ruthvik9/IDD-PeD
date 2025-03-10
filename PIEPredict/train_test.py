"""
The code implementation of the paper:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import os
import sys
import pdb

from PIEPredict.iddped_intent import IDDPIntent
from PIEPredict.iddped_predict import IDDPPredict

# from pie_data import PIE
from PIEPredict.iddped_interface_traj import IDDPedestrian

import keras.backend as K
import tensorflow as tf

from prettytable import PrettyTable

dim_ordering = K.image_dim_ordering()


def train_predict(dataset='iddp',
                  train_test=2, 
                  intent_model_path='data/iddp/intention/context_loc_pretrained'):
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default, test, crosstest
                 'seq_type': 'trajectory',
                #  'time': 'night',
                # 'interaction': 'n',
                # 'sig_type': 'n',
                # 'occluded': 'n',
                 'min_track_size': 60,
                 'random_params': {'ratios': None,
                                 'val_data': True,
                                 'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}

    t = IDDPPredict()
    # pie_path = os.environ.copy()['PIE_PATH']
    iddp_path = os.environ.copy()['IDDP_PATH']

    if dataset == 'pie':
        # imdb = PIE(data_path=pie_path)
        pass
    elif dataset == 'iddp':
        imdb = IDDPedestrian(data_path=iddp_path)
        # pdb.set_trace()

    traj_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['bbox'],
                       'dec_input_type': ['intention_prob', 'obd_speed'],
                       'prediction_type': ['bbox'] 
                       }

    speed_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['obd_speed'], 
                       'dec_input_type': [],
                       'prediction_type': ['obd_speed'] 
                       }

    # traj_model_path = 'data/iddp/trajectory/loc_intent_speed_pretrained'
    # speed_model_path = 'data/iddp/speed/speed_pretrained'

    traj_model_path = 'data/iddp/trajectory/14Sep2024-08h22m03s'
    speed_model_path = 'data/iddp/speed/14Sep2024-09h13m48s'

    if train_test < 2:
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        traj_model_path = t.train(beh_seq_train, beh_seq_val, **traj_model_opts) # cuz it's already trained
        speed_model_path = t.train(beh_seq_train, beh_seq_val, **speed_model_opts)

    if train_test > 0:
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

        perf_final = t.test_final(beh_seq_test,
                                  traj_model_path=traj_model_path, 
                                  speed_model_path=speed_model_path,
                                  intent_model_path=intent_model_path)
        # print("The keys of perf final are",perf_final.keys())

        # t = PrettyTable(['MSE', 'C_MSE'])
        # COMMENTING OUT THE RESULTS TABLES FOR NOW
        # t = PrettyTable(['MSE-15','MSE-30','MSE-45','MSE-last','C-MSE-15','C-MSE-30','C-MSE-45','C-MSE-last'])
        # t.title = 'Trajectory prediction model (loc + IDDP_motive + IDDP_speed)'
        # # t.add_row([perf_final['mse-45'], perf_final['c-mse-45']])
        # t.add_row([perf_final['mse-15'],perf_final['mse-30'],perf_final['mse-45'],perf_final['mse-last'],perf_final['c-mse-15'],perf_final['c-mse-30'],perf_final['c-mse-45'],perf_final['c-mse-last'],])
        
        # print(t)

#train models with data up to critical point
#only for PIE
#train_test = 0 (train only), 1 (train-test), 2 (test only)
def train_intent(train_test=1):

    data_opts = {'fstride': 1,
            'sample_type': 'all', 
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',  #  kfold, random, default
            'seq_type': 'intention', #  crossing , intention
            'min_track_size': 0, #  discard tracks that are shorter
            'max_size_observe': 15,  # number of observation frames
            'max_size_predict': 5,  # number of prediction frames
            'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
            'balance': True,  # balance the training and testing samples
            'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
            'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
            'encoder_input_type': [],
            'decoder_input_type': ['bbox'],
            'output_type': ['intention_binary']
            }


    t = IDDPIntent(num_hidden_units=128,
                  regularizer_val=0.001,
                  lstm_dropout=0.4,
                  lstm_recurrent_dropout=0.2,
                  convlstm_num_filters=64,
                  convlstm_kernel_size=2)

    saved_files_path = ''

    imdb = IDDPedestrian(data_path=os.environ.copy()['IDDP_PATH'])

    pretrained_model_path = 'data/iddp/intention/14Sep2024-02h07m19s'

    # Commented out because training already happened. If you want to train from scratch, uncomment the below while running train_test.py 0 or train_test.py 1
    # if train_test < 2:  # Train
    #     beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
    #     beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

    #     beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    #     beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

    #     saved_files_path = t.train(data_train=beh_seq_train,
    #                                data_val=beh_seq_val,
    #                                epochs=400,
    #                                loss=['binary_crossentropy'],
    #                                metrics=['accuracy'],
    #                                batch_size=128,
    #                                optimizer_type='rmsprop',
    #                                data_opts=data_opts)

        # print("The sequence overlap rate was",data_opts['seq_overlap_rate'])

    if train_test > 0:  # Test
        if saved_files_path == '':
            saved_files_path = pretrained_model_path
        # The code below is commented out so that the evaluation for the intention prediction
        # model is not done.
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        acc, f1 = t.test_chunk(beh_seq_test, data_opts, saved_files_path, False) # Testing it in chunks of 100.
        
        t = PrettyTable(['Acc', 'F1'])
        t.title = 'Intention model (local_context + bbox)'
        t.add_row([acc, f1])
        
        # print(t)

        # K.clear_session()
        # tf.reset_default_graph()
        return saved_files_path

def main(dataset='iddp',train_test=2):

    # intent_model_path = train_intent(train_test=train_test)
    intent_model_path = 'data/iddp/intention/14Sep2024-02h07m19s'
    train_predict(dataset=dataset, train_test=train_test, intent_model_path=intent_model_path)


if __name__ == '__main__':
    try:
        train_test = int(sys.argv[1])
        # vid_num = int(sys.argv[2])
        main(train_test=train_test)
    except ValueError:
        raise ValueError('Usage: python train_test.py <train_test>\n'
                         'train_test: 0 - train only, 1 - train and test, 2 - test only\n')
