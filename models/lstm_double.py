from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
#from gensim.models import KeyedVectors

#word_vectors = KeyedVectors.load_word2vec_format('./embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin', binary=True)

model = Sequential()
model.add(word_vectors.get_keras_embedding(train_embeddings=True))
model.add(LSTM(args.value,
               activation='tanh', # activation function used
               recurrent_activation='hard_sigmoid', # activation function for recurrent step
               use_bias=True, # whether the layer uses a bias vector
               kernel_initializer='glorot_uniform', # initialiser for the weights matrix
               recurrent_initializer='orthogonal', # initialiser for the recurrent kernal's weights
               bias_initializer='zeros', # initialiser for the bias vector
               unit_forget_bias=True, # add 1 to the bias of the forget gate at initialization
               kernel_regularizer=None, # regularizer function applied to kernal
               recurrent_regularizer=None, # regularizer function applied to recurrent kernal
               bias_regularizer=None, # regularizer function applied to bias vector
               activity_regularizer=None, # regularizer function applied to output of the layer
               kernel_constraint=None, # constraint function applied to the kernal
               recurrent_constraint=None, # constraint function applied to the recurrent kernal
               bias_constraint=None, # constraint function applied to the bias vector
               dropout=0.0, # fraction of units to drop for the linear transformation of the inputs
               recurrent_dropout=0.0, # fraction of units to drop for the linear transformation of the recurrent state
               implementation=1, # implementation mode, either 1 or 2.
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False, # If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
               unroll=False)) # whether the network will be unrolled, otherwise a symbolic loop will be used.
model.add(Dense(2,
                activation='tanh',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None))
model.compile(optimizer='Adadelta',
              loss='binary_crossentropy',
              metrics=['acc'],
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None)

#from keras.utils import plot_model
#plot_model(model,show_shapes=True, show_layer_names=False, to_file='model.png')
