import numpy
import keras
from keras.utils.np_utils import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average,Lambda

from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers #keras2
from keras.utils import plot_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.optimizers import *

npratio = 4

class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    user_vecs =Dropout(0.2)(vecs_input)
    user_att = Dense(200,activation='tanh')(user_vecs)
    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model


def get_doc_encoder():
    sentence_input = Input(shape=(30,300), dtype='float32')
    droped_vecs = Dropout(0.2)(sentence_input)

    l_cnnt = Conv1D(400,3,activation='relu')(droped_vecs)
    l_cnnt = Dropout(0.2)(l_cnnt)
    l_cnnt = Attention(20,20)([l_cnnt,l_cnnt,l_cnnt])
    l_cnnt = keras.layers.Activation('relu')(l_cnnt)
    
    droped_rep = Dropout(0.2)(l_cnnt)
    title_vec = AttentivePooling(30,400)(droped_rep)
    sentEncodert = Model(sentence_input, title_vec)
    return sentEncodert

def get_user_encoder():
    news_vecs_input = Input(shape=(50,400), dtype='float32')
    
    news_vecs = Dropout(0.2)(news_vecs_input)
    gru_input = keras.layers.Lambda(lambda x:x[:,-15:,:])(news_vecs)
    vec1 = GRU(400)(gru_input)
    vecs2 = Attention(20,20)([news_vecs]*3)
    vec2 = AttentivePooling(50,400)(vecs2)

    user_vecs2 = Attention(20,20)([news_vecs_input]*3)
    user_vecs2 = Dropout(0.2)(user_vecs2)
    user_vec2 = AttentivePooling(50,400)(user_vecs2)
    user_vec2 = keras.layers.Reshape((1,400))(user_vec2)
        
    user_vecs1 = Lambda(lambda x:x[:,-20:,:])(news_vecs_input)
    user_vec1 = GRU(400)(user_vecs1)
    user_vec1 = keras.layers.Reshape((1,400))(user_vec1)

    user_vecs = keras.layers.Concatenate(axis=-2)([user_vec1,user_vec2])
    vec = AttentivePooling(2,400)(user_vecs)
        
    sentEncodert = Model(news_vecs_input, vec)
    return sentEncodert


def get_model(lr,delta,title_word_embedding_matrix):
    doc_encoder = get_doc_encoder()
    user_encoder = get_user_encoder()
    
    title_word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], 300, weights=[title_word_embedding_matrix],trainable=False)
    
    click_title = Input(shape=(50,30),dtype='int32')
    can_title = Input(shape=(1+npratio,30),dtype='int32')
    
    click_word_vecs = title_word_embedding_layer(click_title)
    can_word_vecs = title_word_embedding_layer(can_title)
    
    click_vecs = TimeDistributed(doc_encoder)(click_word_vecs)
    can_vecs = TimeDistributed(doc_encoder)(can_word_vecs)
    
    user_vec = user_encoder(click_vecs)
    
    scores = keras.layers.Dot(axes=-1)([user_vec,can_vecs]) #(batch_size,1+1,) 
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)     
    
    model = Model([can_title,click_title],logits) # max prob_click_positive
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=SGD(lr=lr,clipvalue = delta), 
                  metrics=['acc'])
    
    news_input = Input(shape=(30,),dtype='int32')
    news_word_vecs = title_word_embedding_layer(news_input)
    news_vec = doc_encoder(news_word_vecs)
    news_encoder = Model(news_input,news_vec)
    
    return model, doc_encoder, user_encoder, news_encoder