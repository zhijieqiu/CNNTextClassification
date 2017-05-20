import numpy as np
import re
import itertools
from collections import Counter
import jieba 

def clipper(txt):
    return jieba.cut(txt)


for i in clipper("我要和你dajia"):
    print(i)




def GenerateVectorOfSentence(sentence,wordDict,maxlen):
    vec = np.zeros((maxlen),dtype=np.int32)
    words = list(clipper(sentence))
    if(maxlen!=None and len(words)>maxlen):
        words=words[0:maxlen]
    for i,word in enumerate(words):
        if word not in wordDict:
            continue
        vec[i]=wordDict[word]
    return vec
def LoadTrainData(fileName):
    f = open(fileName,"r",encoding="utf8")
    trainData=[]
    labels = []
    for line in f:
        tokens = line.split("\t")
        trainData.append(tokens[0])
        labels.append(tokens[1])
    f.close()
    return (trainData,labels)
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
def GenerateWordTable(fileName):
    wordTable = dict({})
    f = open(fileName,"r",encoding="utf8")
    for line in f:
#        line = clean_str(line)
        for token in clipper(line):
            if(token not in wordTable):
                wordTable[token]=len(wordTable)
    return wordTable

def load_data_and_labels2(data_file,split_token='\t',encode="utf8",num_classes=2):
    """
    Loads data from an training data file, which contains traindata and labels
    """
    all_data = list(open(data_file, "r",encoding = encode).readlines())
    data=[]
    labels = []
    for data_line in all_data:
        tokens = data_line.split(split_token)
        data.append(tokens[0])
        t = list(np.zeros((num_classes),dtype=np.int32))
        t[int(tokens[1])]=1
        labels.append(t)
    return data,labels
 
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r",encoding="utf8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r",encoding="utf8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
