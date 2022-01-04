import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time


import numpy as np
import tensorflow as tf
import pickle as pkl
from GAK import tf_gak

 
def clip_tensor(X, eps, norm=np.inf):
    if norm not in [np.inf, 2]:
        raise ValueError('Inadequate norm')

    axis = list(range(1, len(X.get_shape())))  
    avoid_zero_div = 1e-12
  
    if norm == np.inf:
        X = tf.clip_by_value(X, -eps, eps)
    elif norm == 2:
        norm = tf.sqrt(tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(X), axis, keepdims=True)))
        factor = tf.minimum(1., tf.math.divide(eps, norm))
        X = X * factor
    return X


def dtw_differntiable(path, x, y, tf_norm=2):
    """
    Make the optimal path a distance function
    """
    x_path = tf.convert_to_tensor(path[0])
    y_path = tf.convert_to_tensor(path[1])   
    if len(x_path) != len(y_path):
        raise ValueError("Error in DTW path length") 
    else:
        dtw_dist = tf.norm(x[x_path[0]] - y[y_path[0]], ord=tf_norm)
        for i in range(1, len(x_path)):
            dtw_dist = tf.add(dtw_dist, tf.norm(x[x_path[i]] - y[y_path[i]], ord=tf_norm))
    return dtw_dist
    
#CNN Architecture
class cnn_class():
    def __init__(self, name, seg_size, channel_nb, class_nb, arch='1'):
        self.name = name
        self.seg_size = seg_size
        self.channel_nb = channel_nb
        self.class_nb = class_nb
        self.x_holder = []
        self.y_holder = []
        self.y_ =[]
        
        
        if arch=='0':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(20,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 2), strides=2),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        if arch=='1':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(66,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=4),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        elif arch=='2':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(100,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(50,[1, 5], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(50,[1, 3], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 2), strides=1),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(200, activation=tf.nn.relu),
                tf.keras.layers.Dense(100, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        elif arch=='3':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(100,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(50,[1, 6], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(25,[1, 3], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 2), strides=1),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, activation=tf.nn.relu),
                tf.keras.layers.Dense(50, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        self.model = tf.keras.Sequential([self.trunk_model,
            tf.keras.layers.Softmax()])
        #Training Functions
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

                
            
    
    def train(self, train_set, checkpoint_path="TrainingRes/model_target", epochs=10, new_train=False):

        @tf.function
        def train_step(X, y):
            with tf.GradientTape() as tape: 
                pred = self.model(X, training=True)
                pred_loss = self.loss_fn(y, pred)
                total_loss = pred_loss 
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
        if not new_train:
            self.model.load_weights(checkpoint_path)
            sys.stdout.write("\nWeights loaded!")
        else:
            for ep in range(epochs):
                sys.stdout.write("\r{}: Epochs {}/{} . . .".format(self.name, ep+1, epochs))
                sys.stdout.flush()
                for X, y in train_set:
                    train_step(X, y)
                self.model.save_weights(checkpoint_path)
            sys.stdout.write("\n")
                
                
    
    def rots_train(self, train_set, a_shape, 
                   checkpoint_path="TrainingRes/rots_model",
                   gamma_gak=1, gak_sampled_paths=100, path_limit=100,
                   K=100, lbda=1.0, gamma_k=1, eta_k=1e-2, 
                   beta=5e-2, a_init=1e-1, omega=1e-3,
                   new_train=False, X_valid=[], y_valid=[], uses_L2=False,
                   verbose=False):
        model_path = checkpoint_path+'/'+self.name
        self.omega = omega
        
        #decaying l_r of eta_k 
        boundaries = list(np.arange(np.ceil(K/4),K, 1e-2*K))
        values = [eta_k]
        for i, _ in enumerate(boundaries):
            values.append(eta_k/(2**(i+1)))
        lr_schedule_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        self.ro_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule_fn(tf.Variable(0)))
        
        #decaying l_r of gamma_k 
        gamma_k_init = gamma_k
        gamma_k_decay = [gamma_k_init]
        for i in range(1,K):
            if i%10==0:
                gamma_k_init /= 10
            gamma_k_decay.append(gamma_k_init)
            
        def sample_function(input_data):
            rand_batch = np.random.randint(0, len(input_data)-1)
            i=0
            for X, y in input_data:
                if i==rand_batch:
                    return X, y
                else: i+=1
        
        def dist_func(x1, x2, use_log=True, path_limit=path_limit):
            if use_log:
                return -tf.math.log(tf_gak(x1, x2, gamma_gak, path_limit=path_limit))
            else:
                return tf_gak(x1, x2, gamma_gak, path_limit=path_limit)      
        
        @tf.function
        def GW_ro_train_step(X, y, a, lbda):
            with tf.GradientTape() as tape1: 
                pred = self.model(tf.add(X, a), training=True)
                loss_it = self.loss_fn(y, pred)
                G_w = tape1.gradient(loss_it, self.model.trainable_variables)
                if verbose: sys.stdout.write("\n---Current Loss_w:", loss_it)
            return G_w
            
        @tf.function
        def Ga_ro_train_step(X, y, a, lbda):
            with tf.GradientTape() as tape2: 
                tape2.watch(a)
                D_nl = dist_func(X, tf.add(X, a), use_log=False, path_limit=path_limit) #D no log = h_ij
                self.omega = tf.add(tf.multiply(tf.subtract(tf.cast(1, dtype=tf.float64),beta), self.omega), tf.multiply(beta, D_nl))#line 8
                G_omega = tape2.gradient(-tf.math.log(self.omega), a)
                   
            with tf.GradientTape() as tape2: 
                tape2.watch(a)
                pred_a = self.model(tf.add(X, a), training=True)
                loss_it_a = tf.cast(self.loss_fn(y, pred_a), tf.float64)
                G_a_pred = tape2.gradient(loss_it_a, a)
                
            G_a = tf.add(G_a_pred, tf.multiply(lbda, tf.add(G_omega,a)))
            return G_a
        @tf.function
        def Ga_euclidean(X, y, a, lbda):
            with tf.GradientTape() as tape2: 
                tape2.watch(a)
                pred = self.model(tf.add(X, a), training=True)
                loss_it = tf.cast(self.loss_fn(y, pred), tf.float64)
                D = tf.norm(a, ord='euclidean')
                loss_a = tf.add(loss_it, tf.multiply(lbda, D))
                grad_a = tape2.gradient(loss_a, a)
                G_a = tf.add(grad_a, tf.multiply(lbda, a))
            return G_a
            
            
        if not new_train:
            self.model.load_weights(model_path)
            sys.stdout.write("\nWeights loaded!")
        else:
            sys.stdout.write("\nRO training ...")
            #a = tf.Variable(a_init, dtype=tf.float64)
            a = a_init * tf.ones(a_shape, dtype=tf.float64)
            beta= tf.Variable(beta, dtype=tf.float64)
            lbda = tf.cast(lbda,  dtype=tf.float64)            
            min_loss = np.inf
            grads = []
            for k in range(K): 
                t1 = time.time()
                X, y = sample_function(train_set) #line 4 
                G_w = GW_ro_train_step(X, y, a, lbda) #Get the gradient
                self.ro_optimizer.apply_gradients(zip(G_w, self.model.trainable_variables)) #line 6
                if not uses_L2: #line 7
                    G_a = Ga_ro_train_step(X, y, a, lbda)
                else:
                    G_a = Ga_euclidean(X, y, a, lbda)
                    
                a = tf.add(a, tf.multiply(gamma_k_decay[k], G_a)) #line 13
                self.model.save_weights(model_path)                    
                grads.append((G_w, G_a))
                pkl.dump(grads, open("Grads/grad_"+str(path_limit)+self.name+".pkl", "wb"))
                ### Save best weights
                pred_t = self.model(X_valid)
                loss_t = self.loss_fn(y_valid, pred_t)
                if loss_t < min_loss:
                    best_W_T = self.ro_optimizer.get_weights()
                    if verbose: sys.stdout.write("\nBest weight validatio score: {:.2f}".format(self.score(X_valid, y_valid)), flush=True)
                    min_loss = loss_t
                t2 = time.time()
                sys.stdout.write("\nK={}/{} executed in {:.2f} min . . .Score: {:.2f}".format(k+1, K, ((t2-t1)/60), self.score(X, y)), flush=True)
            self.ro_optimizer.set_weights(best_W_T) 
            self.model.save_weights(model_path)
            
    def predict(self, X):
        return tf.argmax(self.model(X, training=False), 1)
        
    def predict_stmax(self, X):
        return self.trunk_model(X, training=False)
    
    def score(self, X, y):
        X = tf.cast(X, tf.float64)
        acc = tf.keras.metrics.Accuracy()
        acc.reset_states()
        pred = self.predict(X)
        acc.update_state(pred, y)
        return acc.result().numpy()
    
    
        
             
        
             