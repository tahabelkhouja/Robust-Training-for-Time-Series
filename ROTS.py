import sys
import os
import json

import tensorflow as tf
import pickle as pkl


from absl import app, flags
from CNNmodel import cnn_class
from hyparamtuning import get_hyparams
FLAGS = flags.FLAGS
def main(argv):
    json_param = "datasets_parameters.json"
    with open(json_param) as jf:
        info = json.load(jf)
        d = info[FLAGS.dataset_name]
        path = d['path']
        SEG_SIZE = d['SEG_SIZE']
        CHANNEL_NB = d['CHANNEL_NB']
        CLASS_NB = d['CLASS_NB']
    #Data Reading
    X_train, y_train, X_test, y_test = pkl.load(open(path, 'rb'))    
    sys.stdout.write("{} - Shape:{}\n".format(FLAGS.dataset_name, X_train.shape))
    #Model Training
    experim_path = "Experiment_"+FLAGS.dataset_name
    try:
        os.makedirs(experim_path)
    except FileExistsError:
        pass
    gamma, FLAGS.rots_lambda = get_hyparams(FLAGS.dataset_name)
    rots_train_path = "{}/TrainingRes/ROTS_lambda_{}_beta_{}".format(experim_path, FLAGS.rots_lambda, FLAGS.rots_beta)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(FLAGS.batch)
    rots_model = cnn_class("ROTS_"+FLAGS.dataset_name, SEG_SIZE, CHANNEL_NB, CLASS_NB, arch='2')
    rots_model.rots_train(train_ds, a_shape=(SEG_SIZE, CHANNEL_NB), gamma_gak=gamma, K=FLAGS.K, path_limit=FLAGS.rots_gak_sample,
                new_train=True, checkpoint_path=rots_train_path,
                gamma_k=5e-2, lbda=FLAGS.rots_lambda, a_init=1e-1, eta_k=1e-2, beta=FLAGS.rots_beta,
                verbose=False)
        
if __name__=="__main__":
    flags.DEFINE_string('dataset_name', 'BME', 'Dataset name')
    flags.DEFINE_integer('batch', 20, 'Batch Size')
    flags.DEFINE_integer('K', 100, 'ROTS Iterations')
    flags.DEFINE_integer('rots_gak_sample', 10, 'ROTS GAK path sampling')
    flags.DEFINE_float('rots_lambda', 10, 'ROTS lambda value')
    flags.DEFINE_float('rots_beta', 5e-2, 'ROTS beta value')
    app.run(main)            