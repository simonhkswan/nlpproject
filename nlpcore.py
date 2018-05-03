import os
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import sys

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences


dl_PATH = './downloads/'

def maybe_download(DATA_URL):

    if not os.path.exists(dl_PATH):
        os.makedirs(dl_PATH)
        print('Dowloads path created.')
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dl_PATH, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        if filename[-3:] == 'zip':
            with zipfile.ZipFile(filepath, 'r') as zipref:
                zipref.extractall(dl_PATH)
                print('Successfully unzipped', filename)
    return()

class TextData(object):

    def __init__(self, xml):
        with open(dl_PATH+xml) as fd:
            self.ETree = ET.parse(fd)

    def totaldocNo(self):
        return(len(self.get_documents()))

    def totsentNo(self):
        N = 0
        for doc in self.getdocuments():
            N += len(doc[2][:])
        return(N)

    def get_docs(self, start=None, stop=None):
        return(self.ETree.getroot()[0][start:stop])

    def tosent(doc):
        return(doc[2][:])

    def get_sentences(self):
        sentences = []
        for doc in self.get_docs():
            for part in doc[1:]:
                for sent in part[:]:
                    sentences.append(sent)
        return(sentences)


def toString(sentElement):

    sent = sentElement.text
    if sent == None:
        sent = ''
    ccuelen = len(sentElement.getchildren())
    if ccuelen > 0:
        for i in range(ccuelen):
            if sentElement[i].tag == 'xcope':
                sent += toString(sentElement[i])
                if sentElement[i].tail != None:
                    sent += sentElement[i].tail
            elif sentElement[i].tag == 'cue':
                sent += sentElement[i].text
                if sentElement[i].tail != None:
                    sent += sentElement[i].tail
    return(sent)


def toStrings(sentElements):

    strings = []
    for element in sentElements:
        strings.append(toString(element))
    return(strings)


def hasSpeculation(sentElement):

    if sentElement.get('certainty')=='uncertain':
        return(True)
    for ele in sentElement.iter():
        if ele.tag == 'cue':
            if ele.attrib['type'] == 'speculation':
                return(True)
    return(False)


def hasNegation(sentElement):

    for ele in sentElement.iter():
        if ele.tag =='cue':
            if ele.attrib['type'] == 'negation':
                return(True)
    return(False)


def get_cues(sentElement):

    return sentElement.getchildren()


def num_words(string):

    return len(text_to_word_sequence(string,
                                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                     lower=True,
                                     split=" "))


def cue_positions(sentElement):

    pos = 0
    positions = []
    pos += num_words(toString(sentElement))
    for cue in get_cues(sentElement)[::-1]:
        pos -= num_words(cue.tail)
        pos -= num_words(cue.text)
        positions.append(pos)
    return(positions[::-1])


def eleList(sentElement):

    elements = []
    if len(sentElement.getchildren()) == 0:
        return([sentElement])
    else:
        elements.extend([sentElement])
        for child in sentElement.getchildren():
            elements.extend(eleList(child))
        return([elements])


def tree_Seperation(ele):

    num = []
    if type(ele) != type([]):
        if ele.text == None:
            num.append(0)
        else:
            #num.append(len(ele.text.split()))
            num.append(len(text_to_word_sequence(ele.text,
                                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                     lower=True,
                                     split=" ")))
        if ele.tail == None:
            num.append(0)
        else:
            #num.append(len(ele.tail.split()))
            num.append(len(text_to_word_sequence(ele.tail,
                                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                     lower=True,
                                     split=" ")))
    else:
        if ele[0].text == None:
            num.append(0)
        else:
            #num.append(len(ele[0].text.split()))
            num.append(len(text_to_word_sequence(ele[0].text,
                                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                     lower=True,
                                     split=" ")))
        for i in range(1,len(ele)):
            num.extend(nestlen(ele[i]))
        if ele[0].tail == None:
            num.append(0)
        else:
            #num.append(len(ele[0].tail.split()))
            num.append(len(text_to_word_sequence(ele[0].tail,
                                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                     lower=True,
                                     split=" ")))
    return(num)


def unravel(ele):

    unrvl = []
    if type(ele) != type([]):
        unrvl.append('<'+ele.tag[:3]+':'+ele.attrib['type'][:3]+':'+ele.attrib['ref'])
        unrvl.append('>'+ele.tag[:3]+':'+ele.attrib['ref'])
    else:
        unrvl.append('<'+ele[0].tag[:3]+':'+ele[0].attrib['id'])

        for i in range(1, len(ele)):
            unrvl.extend(unravel(ele[i]))
        unrvl.append('>'+ele[0].tag[:3]+':'+ele[0].attrib['id'])
    return(unrvl)


def tree_Position(lenlist):

    positions = []
    location = 0
    for n in lenlist:
        positions.append(location)
        location+=n
    return(positions)


def word2index(words, embed_dict):
    # Takes a list of words and indexes them, returning a list of integers.

    indexed = []
    for word in words:
        try:
            indexed.append(embed_dict[word])
        except KeyError:
            indexed.append(embed_dict['UNK'])
    return(indexed)

def import_embedding(location):

    embed_dict={}
    embed_dict['UNK']=0
    with open(location, 'r') as fr:
        data = fr.readlines()
        for line in data:
            mapping = line.split(' ')
            try:
                embed_dict[mapping[0]]=mapping[1]
            except IndexError:
                continue
    return(embed_dict)


def generate_batches(sentences, maxlen, batchsize, embed_dict):
    # Creates a list of input data for training the RNN.
    # Each batch contains sentences with lengths binned into mulitples of 10.

    max_size = int((maxlen+9)/10)
    size_grouped = []
    labels_grouped = []
    for i in range(max_size):
        size_grouped.append([])
        labels_grouped.append([])
    for sentence in sentences:
        string = toString(sentence)
        words = text_to_word_sequence(string,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=" ")
        size = int((len(words)+9)/10)
        if size <= max_size:
            indexed_words = word2index(words,embed_dict)
            speculation = hasSpeculation(sentence)
            certainty = hasNegation(sentence)
            size_grouped[size-1].append(indexed_words)
            labels_grouped[size-1].append([int(certainty),int(speculation)])

    batches = []
    validation_batches = []

    l = 0
    for i in range(len(size_grouped)):
        l+=10
        rd.seed(447)
        rd.shuffle(size_grouped[i])
        rd.seed(447)
        rd.shuffle(labels_grouped[i])

        numbatch = int(len(size_grouped[i])/batchsize) #currently misses last few batches
        for j in range(numbatch):
            padded = pad_sequences(size_grouped[i][(j)*batchsize:(j+1)*batchsize],
                                   maxlen=l,
                                   dtype='int32',
                                   padding='pre',
                                   truncating='pre',
                                   value=0)
            if j%5 == 1:
                validation_batches.append([padded, np.array(labels_grouped[i][(j)*batchsize:(j+1)*batchsize])])
            else:
                batches.append([padded, np.array(labels_grouped[i][(j)*batchsize:(j+1)*batchsize])])

    return(batches,validation_batches)


def conf_matrix(y_true, y_pred, title='Confusion Matrix', threshold=50, filename=None, display=False):
    activation = np.linspace(0,1,50)
    f_max = 0
    fs = []
    a_max = 0
    for a in activation:
        y_act = np.where(y_pred>a, 1, 0)
        cm = confusion_matrix(y_true,y_act)
        cm2 = normalize(cm,axis=1,norm='l1')
        p = float(cm2[0][0])
        r = float(cm2[1][1])
        f = 2*p*r/(p+r)
        fs.append(f)
        if f > f_max:
            f_max = f
            a_max = a
    Fs = np.array(fs)

    y_act = np.where(y_pred>a_max, 1, 0)
    cm = confusion_matrix(y_true,y_act)
    cm2 = normalize(cm,axis=1,norm='l1')
    p = cm2[0][0]
    r = cm2[1][1]
    f = 2*p*r/(p+r)

    fig3 = plt.figure(figsize=(8,8))
    ax3 = fig3.gca()
    res = ax3.imshow(np.array(cm2*100), cmap=plt.cm.RdPu)
    width, height = cm.shape
    threshold = 0.5

    for x in range(width):
        for y in range(height):
            if cm2[x][y] < threshold: col = [0,0,0]
            else: col = [1,1,1]
            ax3.annotate('%d\n(%.1f%%)'%(cm[x][y],cm2[x][y]*100), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',color=col)

    cb = fig3.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title+': F = %.2f A = %.2f'%(f,a_max))
    if type(filename) == type('a'):
        plt.savefig(filename,dpi=300)
    if display:
        plt.show()

    fig4 = plt.figure(figsize=(8,5))
    ax4 = fig4.gca()
    ax4.plot(activation,Fs,color=[0.65,0.1,0.18])
    plt.ylabel('F-Score')
    plt.xlabel('Activation')
    plt.savefig(filename[:-4]+'_Fs.png',dpi=300)
