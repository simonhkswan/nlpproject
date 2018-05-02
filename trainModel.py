import argparse
from nlpcore import*
import tensorflow as tf
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint,Callback
from keras import backend as K
from gensim.models import KeyedVectors
import csv


parser = argparse.ArgumentParser(description='Train a selected model and save the training log.')
parser.add_argument('model_location', metavar='m', type=str)
parser.add_argument('logs_dest', metavar='l', type = str)
parser.add_argument('embedding_location', metavar='e', type = str)
args = parser.parse_args()

dl=PATH = './downloads/'
if not os.path.exists(args.logs_dest):
    os.makedirs(args.logs_dest)

maybe_download('http://rgai.inf.u-szeged.hu/project/nlp/bioscope/bioscope.zip')
maybe_download('http://rgai.inf.u-szeged.hu/~vinczev/conll2010st/task2_eval.zip')

data = TextData('abstracts.xml')
data2 = TextData('full_papers.xml')

sentences1 = data.get_sentences()
sentences2 = data2.get_sentences()
sentences = sentences1 + sentences2

print('%d training sentences loaded'%(len(sentences)))

print('Loading word vectors.')
word_vectors = KeyedVectors.load_word2vec_format(args.embedding_location, binary=True)
print(args.embedding_location+' loaded.')
word_vectors.save_word2vec_format(args.embedding_location, fvocab=args.embedding_location[:-4]+'_vocab.txt', binary=True)
print('Embedding mapping saved.')
embed_dict=import_embedding(args.embedding_location[:-4]+'_vocab.txt')
print('Embedding dictionary loaded, %d vectors in total.'%(len(embed_dict)))

batches,batches2 = generate_batches(sentences,80,10,embed_dict)
train=batches[:]
test=batches2[:]
rd.shuffle(test)
rd.shuffle(train)

print('Batches generated: %d training batches, %d validation batches.'%(len(train),len(test)))

#import the desired model
exec(compile(source=open(args.model_location).read(),filename=args.model_location,mode='exec'))

#model.load_weights('./logs/wordvec_model.h5',by_name=True)
model.summary()
#plot_model(model, to_file='./images/modalityLSTMmodel.png', show_shapes=True)
model_checkpoint = ModelCheckpoint(args.logs_dest+'/model.h5')

epoch=1

logOUT = []
for i in range(10):
    batchNO=0
    for batch in train:
        if batchNO%200 == 0:
            vloss,vacc,batch2tot = 0,0,0
            vacc_best = 0
            VY = []
            PY = []
            for batch2 in test:
                #print(batch2)
                [vx,vy] = batch2
                [val_loss,val_acc] = model.test_on_batch(vx,vy)
                vloss += val_loss
                vacc += val_acc
                batch2tot += 1

                VY.append(vy)
                py = model.test_on_batch(vx)
                PY.append(py)

            vloss = vloss/batch2tot
            vacc = vacc/batch2tot
            print('Batch number: %d.%d Validation accuracy: %f Validation loss: %f'%(epoch, batchNO, vacc, vloss))

            if vacc > vacc_best:
                print('Saving model weights...')
                model.save_weights(args.logs_dest+'model.h5')
                vacc_best = vacc

            print('Generating Confusion Matrix')
            corr_Y = np.concatenate(VY, axis = 0)
            pred_Y = np.concatenate(PY, axis = 0)
            conf_matrix(corr_Y[:,0], pred_Y[:,0], filename = args.logs_dest+'confmatrix/epoch%2d.png'%(epoch))

        [x,y] = batch
        [loss, acc] = model.train_on_batch(x,y)

        logOUT.append([epoch,batchNO,loss,acc,vloss,vacc])
        batchNO +=1
    epoch += 1

print('Saving training logs...')
with open(args.logs_dest+'modelLOG.csv', 'w') as logFile:
    writer = csv.writer(logFile)
    writer.writerows(logOUT)

print("Done.")
