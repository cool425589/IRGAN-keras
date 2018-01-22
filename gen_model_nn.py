import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import tensorflow as tf
from keras.activations import softmax
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers import concatenate
from keras.layers.core import Reshape, Dense, Dropout, Flatten, Activation, Lambda, Permute
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import optimizers
class GEN:

    def __init__(self,feature_size, hidden_size, weight_decay, learning_rate, temperature=1.0, layer_0_param = None, layer_1_param = None):

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.pred_data = Input(shape=(None, self.feature_size)) 

        self.reward = Input(shape=(None,)) 
        self.important_sampling = Input(shape=(None,)) 
 

        self.Dense_1_result = Dense(self.hidden_size, input_dim = self.feature_size, activation='tanh', weights = layer_0_param, kernel_regularizer = regularizers.l2(self.weight_decay))(self.pred_data)
        self.Dense_2_result = Dense(1, input_dim = self.hidden_size, weights = layer_1_param, kernel_regularizer = regularizers.l2(self.weight_decay))(self.Dense_1_result)        
        self.score = Lambda(lambda x: x / self.temperature)(self.Dense_2_result)
        self.score = Reshape([-1])(self.score)
        self.prob = Activation('softmax')(self.score) 
        self.model = Model(inputs = [self.pred_data, self.reward, self.important_sampling ], outputs = [self.prob])
        #self.__build_train_fn()
        self.model.summary()
        self.model.compile(loss = self.loss(self.reward, self.important_sampling), optimizer = optimizers.TFOptimizer(tf.train.GradientDescentOptimizer(self.learning_rate)), metrics=['accuracy'])
    def loss(self, _reward, _important_sampling):
        def _loss(y_true, y_pred):
            log_action_prob = K.log(y_pred)           
            loss = - K.reshape(log_action_prob,[-1])*K.reshape(_reward,[-1])*K.reshape(_important_sampling,[-1])
            loss = K.mean(loss)
            return loss
        return _loss    
    def get_score(self, pred_data):
        functor  = K.function([self.model.layers[0].input]+[ K.learning_phase()], [self.model.layers[4].output])
        layer_outs  = functor ([ pred_data, 0.])
        return layer_outs 
    def get_prob(self, pred_data):
        functor  = K.function([self.model.layers[0].input]+[ K.learning_phase()], [self.model.layers[5].output])
        layer_outs  = functor ([ pred_data, 0.])
        return layer_outs 
    """
    def __build_train_fn(self):

        #self.model.train_on_batch([pred_data, gan_prob, reward.reshape([-1]), important_sampling], [score.reshape([-1])])
        action_important_sampling_placeholder =  K.placeholder(shape=(None,), name="action_onehot")
        action_prob_placeholder = self.model.output
        discount_reward_placeholder = K.placeholder(shape=(None,) , name="discount_reward")
        discount_sample_index_placeholder = K.placeholder(shape=(None,) , name="discount_reward", dtype = 'int32')
        gan_prob = K.gather(K.reshape(action_prob_placeholder,[-1]), discount_sample_index_placeholder)
        log_action_prob = K.log(gan_prob)
        
        loss = - log_action_prob * discount_reward_placeholder * action_important_sampling_placeholder
        loss = K.mean(loss)

        adam = optimizers.Adam(lr = self.learning_rate)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           discount_sample_index_placeholder,
                                           action_important_sampling_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)
    def train(self, pre_data, sample_index, reward, important_sampling):
        self.train_fn([pre_data, sample_index, reward, important_sampling])
    """
    def train(self, pre_data, reward, important_sampling):
        self.model.train_on_batch([pre_data, reward, important_sampling], np.zeros([pre_data.shape[0]]))

    def save_model(self, filename):
        self.model.save_weights(filename)

