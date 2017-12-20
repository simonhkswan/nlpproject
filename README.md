# Applying new methods for relation extraction and negation to biomedical text mining

With the large volume of biomedical information published on a day-to-day basis, there is an increasing demand for the application of natural language processing techniques to literature-based discovery tools to assist in extracting information on diseases, genes, proteins and other biological concepts.
There has been considerable development in machine learning models for natural language processing though many of these techniques have not yet been directly applied to biomedical text mining.
The aim of this project is to incorporate some of these new methods to improve the quality and reliability of information extracted in biomedical publications. The two tasks that will be speciﬁcally explored are modality/negation identiﬁcation in text and relationship extraction - the detection and classiﬁcation of semantic relationships between various mentions of interest.
This preliminary submission ﬁrst provides a brief overview of text mining in relation to literature based discovery as a background introduction to the goals of the project. The speciﬁc points of focus of the project are then detailed in the following section and an overall project plan concludes the report.

## 1. Introductory Background

Text mining is the process by which high-quality structured information is distilled from (usually unstructured) text bodies. When applied to scientiﬁc literature, text mining tools becoming increasingly used to assist extracting new information, often in a semi-automated way. The use of such Literature-Based-Discovery (LBD) tools has already led to many discovery proposals such as potential treatments for Parkinson’s disease [8] and for cataracts [7].
Most LBD tools are derived from Swanson’s ABC co-occurrence model [12]. Text mining is used to extract A-B and B-C relationships. New knowledge is then ‘discovered’ by concluding the implicit A-C relationship. With such high volumes of text published every day [5], islands of unconnected knowledge depend on LBD to be linked together.
It is therefore the purpose of text mining, in the scope of LBD, to ﬁnd terms (‘A’, ‘B’, ‘C’) and determine some form of relationship. To handle this task, three important questions are raised:

1. How are terms (words, phrases, etc) represented?
2. What constitutes a relationship and how do we ﬁnd this relationship?
3. How do we evaluate the performance of the text-mining models used?

### 1.1. Term representation

In natural language processing, words are often tokenized for ease of handling in computer programs. Terms which are uninformative (overly general, extremely common) are often removed; this improves the quality of extracted relationships as well as the speed of the program. Lists of such general words (stop words such as ‘the’, ‘a’) and words uninteresting to the biomedical domain are used to simplify the text inputs of models.
Various semantic type ﬁlters are also used to help label words of interest. Within the biomedical literature, one such notable ﬁlter uses the Genia Ontology [6], which allows for the categorization of important words within a document, e.g. tags such as ‘protein complex’ and ‘DNA domain or region’ help to identify scientiﬁc terminology. Words signifying important relationships are also categorized to help standardize the relationships extracted, e.g. ‘positive regulation’, ‘DNA modiﬁcation’.
Extremely important to extracting information and concepts from the literature is the appreciation that the symbolic nature of text should be considered as a higherlevel representation of how information is conceived in our brains where ideas, words and concepts are carried through a connectionist representation. In this three level model of human cognition developed by Ga¨rdenfors [3], It is this intermediate-level conceptual space that is more directly applicable to scientiﬁc reasoning and abduction [1].
A conceptual space can be perceived as a high dimensional space where speciﬁc properties have a geometric representation. Words and terms in a text can then be considered as vectors within this space mapping to a particular point. Similar words would be found in similar locations. And so, by instead embedding labelled words, terms, and sentences into a multi-dimensional space, the gap between cognitive knowledge representation and actual computation representations can be bridged – NLP programs can process text in a more similar manner to the human brain. This is known as the Hyperspace Analogue to Language model [9].

Many forms of embedding are used as standard practise in natural language processing and methods of constructing these relational hyperspaces are constantly being explored, with notable embedding methods such as word2vec [10] drastically improving the performance of natural language processing tools. The choice and style of embedding has a signiﬁcant impact on the performance of text-mining and is something which will be explored within this project.

### 1.2. Relationship between terms

Relationships were traditionally deﬁned through co-occurrence of two terms within a sentence. Such relationships can only be interpreted as associations. More complicated semantic models are becoming more common which allows for labelled relationships to be extracted, providing much more meaningful information [4]. This project’s main purpose is to investigate such models to improve the details of extracted relationships in order assist developing co-occurrence models into more complicated ones. The exact nature of how the relationships’ details are to be improved is discussed in the next section.

### 1.3. Evaluation of models

Two important metrics of evaluation in information extraction are precision and recall. Precision refers to the fraction of correct relationships amongst all the relationships extracted, where as recall is the fraction of correct relationships found compared to the total number of relevant relationships that were extractable. These two terms are summarized in the diagram below.

Co-occurrence models typically have a higher recall [4], though the information extracted is typically far less detailed and as a result these models have much worse precision. By extracting more detailed relationships, the information extracted becomes more reliable and meaningful.
The harmonic mean of these two values is known as the F-measure of the model. Many shared tasks and text-mining competitions use F-measures to compare the performance of submitted models.
There exist many large annotated corpora of biomedical text documents which allow for the testing and evaluation of diﬀerent tasks in text mining. The BioScope corpus [13] contains over 20,000 sentences from biological papers annotated for negative and speculative keywords as well as the scope to which they apply. Another important corpus is presented in the cancer and genetics BioNLP Shared Task of 2013 [11]. This corpus covers multiple subdomains of cancer biology and has been labelled with 40 diﬀerent types of relations that are extractable. With these two datasets, different models for detecting negation, uncertainty (modality), and relationship types can be evaluated and compared with baselines from models that already exist.

## 2. Project Goals

With the relevant background concepts explained, the goals of this project can be summarized into improving relationship extraction from simple co-occurrence models in three different fronts: Identifying uncertainty/modality of relationships, detecting the negation of relationships, and classifying the type of relationship (going from simple association to a speciﬁcally deﬁned relationship).
Newly developed neural models and embedding techniques such as incorporating a structured self-attentive sentence embedding are to be applied to the BioScope and BioNLP-ST (2013) corpora and the results will be compared to baselines with the intention of producing improved performance.

## 3. Project Plan

Within the three improvements to relationship extraction explored in this project, datasets have been identiﬁed to develop and test models on:
Modality Identiﬁcation
CoNLL-2010 Shared Task (uses the BioScope Corpus) [2]
Negation Identiﬁcation
BioScope Corpus
Relationship extraction
BioNLP Shared Task 2013 (Cancer and Genetics)

The ﬁrst initial task will be establishing baseline results for simple neural models that have been developed recently, such as a simple CNN (convolution neural network) and an RNN (recurrent neural network). These results will be compared to the performances of models that already exist.
The project will then aim to improve the results of the neural models, speciﬁcally by investigating the use of a newly developed sentence embedding amongst other modiﬁcations to the simple neural models.
The development, training and testing of models will be done using the TensorFlow and Keras packages in Python for their speed in prototyping.

