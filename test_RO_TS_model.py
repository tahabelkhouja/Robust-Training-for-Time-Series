import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import json

import pickle as pkl


from absl import app, flags
from CNNmodel import cnn_class
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
    _, _, X_test, y_test = pkl.load(open(path, 'rb'))    
    #Model Training
    experim_path = "Experiments/Experiment_"+FLAGS.dataset_name
        
    rots_model = cnn_class("ROTS_"+FLAGS.dataset_name, SEG_SIZE, CHANNEL_NB, CLASS_NB, arch='2')
    rots_train_path = "{}/TrainingRes/ROTS_lambda_{}_beta_{}".format(experim_path, FLAGS.rots_lambda, FLAGS.rots_beta)
    rots_model.rots_train([],[],10, checkpoint_path=rots_train_path, new_train=False)
    score = rots_model.score(X_test, y_test)
    sys.stdout.write("\nPerformance of {} ROTS training: {:.2f} on test data\n".format(FLAGS.dataset_name, score))
if __name__=="__main__":
    flags.DEFINE_string('dataset_name', 'SyntheticControl', 'Dataset name')
    flags.DEFINE_float('rots_lambda', -1, 'ROTS lambda value')
    flags.DEFINE_float('rots_beta', 5e-2, 'ROTS beta value')
    app.run(main)            
