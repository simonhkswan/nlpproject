from nlpcore import *

maybe_download('http://rgai.inf.u-szeged.hu/~vinczev/conll2010st/task1_bio_eval.zip')
maybe_download('http://rgai.inf.u-szeged.hu/~vinczev/conll2010st/task2_bio_eval.zip')
maybe_download('http://rgai.inf.u-szeged.hu/project/nlp/bioscope/bioscope.zip')

data = TextData('full_papers.xml')
data2 = TextData('task1_bio_eval.xml')

sentences = data.get_sentences()
sentence_lengths = []
negation_sentences = []
speculation_sentences = []
total = 0
neg_total = 0
spec_total = 0
for sentence in sentences:
    num = num_words(toString(sentence))
    sentence_lengths.append(num)
    if hasNegation(sentence):
        negation_sentences.append(num)
        neg_total += 1
    if hasSpeculation(sentence):
        speculation_sentences.append(num)
        spec_total +=1
    total += 1
print(spec_total,neg_total,total)

fig1 = plt.figure(figsize=(7,3))
ax1 = fig1.gca()
ax1.set_title('Speculation Sentences')
ax1.set_ylabel('Number of sentences')
ax1.set_xlabel('Number of words')
plt.hist(speculation_sentences,bins=81,range=(0,80),color=[0.14,0.5,0.58])
plt.tight_layout()
plt.show()
plt.savefig('./images/FullpapersSpecultionHist.png',dpi=300)
