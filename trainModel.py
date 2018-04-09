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
maybe_download('http://rgai.inf.u-szeged.hu/~vinczev/conll2010st/task2_eval.zip')

data = TextData('abstracts.xml')
data2 = TextData('full_papers.xml')

sentences1 = data.get_sentences()
sentences2 = data2.get_sentences()
sentences = sentences1 + sentences2

print('%d training sentences loaded'%(len(sentences)))

print('Loading word vectors.')
word_vectors = KeyedVectors.load_word2vec_format('./downloads/PubMed-shuffle-win-2.bin', binary=True)
print('PubMed-shuffle-win-2.bin loaded.')
word_vectors.save_word2vec_format('./downloads/PubMed-shuffle-win-2.bin', fvocab='./downloads/PubMed-shuffle-win-2_vocab.txt', binary=True)
print('Embedding mapping saved.')
embed_dict=import_embedding('./downloads/PubMed-shuffle-win-2_vocab.txt')
print('Embedding dictionary loaded, %d vectors in total.'%(len(embed_dict)))

batches,batches2 = generate_batches(sentences,80,10,embed_dict)
train=batches[:]
test=batches2[:]
rd.shuffle(test)
rd.shuffle(train)

print('Batches generated: %d training batches, %d validation batches.'%(len(train),len(test)))

exec(compile(open('./models/lstm_double.py')))

#model.load_weights('./logs/wordvec_model.h5',by_name=True)
model.summary()
#plot_model(model, to_file='./images/modalityLSTMmodel.png', show_shapes=True)
model_checkpoint = ModelCheckpoint('./logs/LSTMmodel.h5')

epoch=1
#TC.set_model(model)
#TC.validation_data=(vX,vY)
logOUT = []
for i in range(10):
    batchNO=0
    for batch in train:
        if batchNO%200 == 0:
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
            print('Batch number: %d.%d Validation accuracy: %f Validation loss: %f'%(epoch, batchNO, vacc, vloss))
        [x,y] = batch
        [loss, acc] = model.train_on_batch(x,y)

        logOUT.append([epoch,batchNO,loss,acc,vloss,vacc])
        batchNO +=1
    epoch += 1

print('Saving training logs...')
with open('./logs/LSTMmodelLOG.csv', 'w') as logFile:
    writer = csv.writer(logFile)
    writer.writerows(logOUT)

print('Generating Confusion Matrix')


print('Saving model weights...')
model.save_weights('./logs/LSTMmodel.h5')


#TC.on_train_end(_)
