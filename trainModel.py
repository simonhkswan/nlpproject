from nlpcore import*
import tensorflow as tf
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
from keras.callbacks import TensorBoard,ModelCheckpoint,Callback
from keras import backend as K

dl=PATH = './downloads/'

maybe_download('http://rgai.inf.u-szeged.hu/~vinczev/conll2010st/task1_train_bio_rev2.zip')
data = TextData('task1_train_bio_abstracts_rev2.xml')
