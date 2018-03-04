from nlpcore import*
import tensorflow as tf
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
from keras.callbacks import TensorBoard,ModelCheckpoint,Callback
from keras import backend as K
from gensim.models import KeyedVectors
import csv

dl=PATH = './downloads/'

maybe_download('http://rgai.inf.u-szeged.hu/project/nlp/bioscope/bioscope.zip')
maybe_download('https://nofile.io/f/PdTE3n32qNr/PubMed-shuffle-win-2.bin')
data = TextData('abstracts.xml')
data2 = TextData('full_papers.xml')

sentences = data.get_sentences()
sentences2 = data2.get_sentences()
print('%d training sentences loaded, %d validation sentences loaded.'%(len(sentences), len(sentences2)))

print('Loading word vectors.')
word_vectors = KeyedVectors.load_word2vec_format('./downloads/PubMed-shuffle-win-2.bin', binary=True)
print('PubMed-shuffle-win-2.bin loaded.')
word_vectors.save_word2vec_format('./downloads/PubMed-shuffle-win-2.bin', fvocab='./downloads/PubMed-shuffle-win-2_vocab.txt', binary=True)
print('Embedding mapping saved.')
embed_dict=import_embedding('./downloads/PubMed-shuffle-win-2_vocab.txt')
print('Embedding dictionary loaded, %d vectors in total.'%(len(embed_dict)))

batches = generate_batches(sentences,80,10,embed_dict)
batches2 = generate_batches(sentences2,80,10,embed_dict)
train=batches[:]
test=batches2[:]
rd.shuffle(test)
rd.shuffle(train)

batch_size = 10

model = Sequential()
model.add(word_vectors.get_keras_embedding())
model.add(LSTM(30,
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
model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['acc'],
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None)
#model.load_weights('./logs/wordvec_model.h5',by_name=True)
model.summary()
#plot_model(model, to_file='./images/modalityLSTMmodel.png', show_shapes=True)

#TC = TensorBoard(log_dir='./logs/LSTM_Neg_Spec', batch_size=batch_size,
#                          histogram_freq=0, write_images=True,
#                          write_grads=False, write_graph=True, embeddings_freq=1)

model_checkpoint = ModelCheckpoint('./logs/LSTMmodel.h5')

epoch=0
batchNO=0
#TC.set_model(model)
#TC.validation_data=(vX,vY)
logOUT = []
for i in range(10):
    vloss,vacc,batch2tot = 0,0,0
    for batch2 in test:
        #print(batch2)
        [vx,vy] = batch2
        [val_loss,val_acc] = model.test_on_batch(vx,vy)
        vloss += val_loss
        vacc += val_acc
        batch2tot += 1
    vloss = vloss/batch2tot
    vacc = vacc/batch2tot

    for batch in train:
        [x,y] = batch
        [loss, acc] = model.train_on_batch(x,y)
        batchNO += 1
        logOUT.append([epoch,batchNO,loss,acc,vloss,vacc])
        if batchNO%200 == 0:
            print('Batch number: %d Validation accuracy: %f Validation loss: %f'%(batchNO, val_acc, val_loss))
    epoch += 1

print('Saving training logs...')
with open('./logs/LSTMmodelLOG.csv', 'w') as logFile:
    writer = csv.writer(logFile)
    writer.writerows(logOUT)
print('Saving model weights...')
model.save_weights('./logs/LSTMmodel.h5')


#TC.on_train_end(_)
