from action_predict import action_prediction
from pie_data import PIE
from PCIP.iddped_interface import IDDPedestrian
import os
import sys
import yaml
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpu,
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    # )

def test_model(saved_files_path=None):

    with open(os.path.join(saved_files_path, 'configs.yaml'), 'r') as yamlfile:
        opts = yaml.safe_load(yamlfile)
    print(opts)
    model_opts = opts['model_opts']
    data_opts = opts['data_opts']
    net_opts = opts['net_opts']

    tte = model_opts['time_to_event'] if isinstance(model_opts['time_to_event'], int) else \
                model_opts['time_to_event'][1]
    data_opts['min_track_size'] = model_opts['obs_length'] + tte
    iddp_path = "data/IDDPedestrian/"
    pie_path = "data/pie/"
    if model_opts['dataset'] == 'pie':
            # pass
            imdb = PIE(data_path=pie_path)
            # imdb.get_data_stats()
    elif model_opts['dataset'] == 'jaad':
            # imdb = JAAD(data_path=os.environ.copy()['JAAD_PATH'])
            imdb = JAAD(data_path='/home/haolin/CITR/PedestrianActionBenchmark/JAAD/')
    elif model_opts['dataset'] == 'iddp':
          imdb = IDDPedestrian(data_path=iddp_path)
    else:
            raise ValueError("{} dataset is incorrect".format(model_opts['dataset']))

    method_class = action_prediction(model_opts['model'])(**net_opts)
    if 'test_image_set' in data_opts.keys() and data_opts['data_split_type'] != 'default':
        # Remember to change data_split_type to 'test'
        beh_seq_test = imdb.generate_data_trajectory_sequence(data_opts['test_image_set'], **data_opts)
    else:
        # Note: To get post-crossing tracks, change seq_type to 'post_crossing' in data_opts
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    # Call the following method to print out the metrics for testing data within each sub-categories.
    # method_class.test_categories(beh_seq_test,'group',saved_files_path) # tte, vehicle_speed, group, post_cross
    method_class.test(beh_seq_test,saved_files_path)
    print('Testing completed...\n')
    # Commented the below line for category wise accuracies.
    # print(acc, auc, f1, precision, recall)


if __name__ == '__main__':
    saved_files_path = sys.argv[1]
    test_model(saved_files_path=saved_files_path)
