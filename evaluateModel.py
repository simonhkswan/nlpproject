import argparse
from nlpcore import*
import tensorflow as tf
from keras import backend as K
from gensim.models import KeyedVectors

parser = argparse.ArgumentParser(description='Evaluate a selected model and save the results.')
parser.add_argument('model_location', metavar='m', type=str)
parser.add_argument('logs_dest', metavar='l', type = str)
parser.add_argument('embedding_location', metavar='e', type = str)
args = parser.parse_args()

maybe_download('http://rgai.inf.u-szeged.hu/~vinczev/conll2010st/task2_eval.zip')

data = TextData('task2_eval.xml')
sentences = data.get_sentences()
print('%d training sentences loaded'%(len(sentences)))

print('Loading word vectors.')
word_vectors = KeyedVectors.load_word2vec_format(args.embedding_location, binary=True)
print(args.embedding_location+' loaded.')
word_vectors.save_word2vec_format(args.embedding_location, fvocab=args.embedding_location[:-4]+'_vocab.txt', binary=True)
print('Embedding mapping saved.')
embed_dict=import_embedding(args.embedding_location[:-4]+'_vocab.txt')

print('Embedding dictionary loaded, %d vectors in total.'%(len(embed_dict)))


#import the desired model
exec(compile(source=open(args.model_location).read(),filename=args.model_location,mode='exec'))

model.load_weights(args.logs_dest+'model.h5',by_name=True)
print('Loaded model weights.')

batches1,batches2 = generate_batches(sentences,80,10,embed_dict)
batches1.extend(batches2)

VY = []
PY = []
for batch in batches1:
    #print(batch2)
    [vx,vy] = batch

    VY.append(vy)
    py = model.predict_on_batch(vx)
    PY.append(py)


print('Generating Confusion Matrix')
corr_Y = np.concatenate(VY, axis = 0)
pred_Y = np.concatenate(PY, axis = 0)

if not os.path.exists(args.logs_dest+'confmatrix/'):
    os.makedirs(args.logs_dest+'confmatrix/')
neg_f, neg_a = conf_matrix(corr_Y[:,0], pred_Y[:,0], filename = args.logs_dest+'confmatrix/neg_validation.png'%(epoch,batchNO))
spec_f, spec_a = conf_matrix(corr_Y[:,1], pred_Y[:,1], filename = args.logs_dest+'confmatrix/spec_validation.png'%(epoch,batchNO))
