import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
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
    sys.stdout.write("{} - Shape:{}".format(FLAGS.dataset_name, X_train.shape))
    #Model Training
    experim_path = "Experiments/Experiment_"+FLAGS.dataset_name
    try:
        os.makedirs(experim_path)
    except FileExistsError:
        pass
    if FLAGS.rots_lambda==-1:
        gamma, FLAGS.rots_lambda = get_hyparams(FLAGS.dataset_name)
    else:
        gamma, _ = get_hyparams(FLAGS.dataset_name)
        
    
    rots_train_path = "{}/TrainingRes/ROTS_lambda_{}_beta_{}".format(experim_path, FLAGS.rots_lambda, FLAGS.rots_beta)
    rots_model = cnn_class("ROTS_"+FLAGS.dataset_name, SEG_SIZE, CHANNEL_NB, CLASS_NB, arch='2')    
    
    if FLAGS.save_with_valid:
        validation_size = int(0.1*X_train.shape[0])
        X_valid = X_train[-validation_size:]
        y_valid = y_train[-validation_size:]
        total_iter = (X_train[:-validation_size].shape[0] * FLAGS.K)//FLAGS.batch + 1
        train_ds = tf.data.Dataset.from_tensor_slices((X_train[:-validation_size], y_train[:-validation_size]))\
                    .shuffle(X_train[:-validation_size].shape[0]).repeat(FLAGS.K).batch(FLAGS.batch, drop_remainder=True)
        rots_model.rots_train(train_ds, (SEG_SIZE, CHANNEL_NB), total_iter, gamma_gak=gamma, path_limit=FLAGS.rots_gak_sample, gak_random_kill=FLAGS.gak_random_kill,
                new_train=True, checkpoint_path=rots_train_path,
                X_valid=X_valid, y_valid=y_valid,
                gamma_k=5e-2, lbda=FLAGS.rots_lambda,  beta=FLAGS.rots_beta,
                verbose=False)
    else:
        total_iter = (X_train.shape[0] * FLAGS.K)//FLAGS.batch + 1
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
                    .shuffle(X_train.shape[0]).repeat(FLAGS.K).batch(FLAGS.batch, drop_remainder=True)
        rots_model.rots_train(train_ds, (SEG_SIZE, CHANNEL_NB), total_iter, gamma_gak=gamma, path_limit=FLAGS.rots_gak_sample, gak_random_kill=FLAGS.gak_random_kill,
                new_train=True, checkpoint_path=rots_train_path,
                gamma_k=5e-2, lbda=FLAGS.rots_lambda,  beta=FLAGS.rots_beta,
                verbose=False)
        
if __name__=="__main__":
    flags.DEFINE_string('dataset_name', 'SyntheticControl', 'Dataset name')
    flags.DEFINE_integer('batch', 11, 'Batch Size')
    flags.DEFINE_integer('K', 10, 'RO-TS Iterations')
    flags.DEFINE_integer('rots_gak_sample', 20, 'RO-TS GAK path sampling')
    flags.DEFINE_integer('gak_random_kill', 5, 'RO-TS GAK path sampling random elimination')
    flags.DEFINE_float('vs', 0.1, 'Validation ratio from training')
    flags.DEFINE_float('rots_lambda', -1, 'RO-TS lambda value')
    flags.DEFINE_float('rots_beta', 5e-2, 'RO-TS beta value')
    flags.DEFINE_boolean('save_with_valid', False, 'Save best weight using validation set')
    app.run(main)            
