from keras.models import Sequential
from keras.layers import Embedding,Conv1D,MaxPooling1D,Dense
#from gensim.models import KeyedVectors

#word_vectors = KeyedVectors.load_word2vec_format('./embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin', binary=True)


model = Sequential()
model.add(word_vectors.get_keras_embedding(train_embeddings=True))
model.add(Conv1D(30, 5, strides=1,
                padding='same', dilation_rate=1,
                activation=None, use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None,
                kernel_constraint=None, bias_constraint=None))
model.add(MaxPooling1D(pool_size=30))
model.add(Dense(2))
model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['acc'],
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None)

#from keras.utils import plot_model
#plot_model(model,show_shapes=True, show_layer_names=False, to_file='model3.png')
