import re
import math

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def read_file(filename):
    file = open(filename, 'r', encoding="utf8")
    content = []
    for line in file.readlines():
        content.append(line)
    return content


def split_data(content):
    labels = []
    data = []
    for i in content:
        row = i.split('\t')
        if len(row) == 2:
            labels.append(row[0])
            data.append(row[1])
    return labels, data
 

#获取单词词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None



def data_profile(data):
    rule = re.compile("[^a-zA-Z ]")
    for i in range(len(data)):
        #转换为小写，并分割
        data[i] = re.sub(rule, '', data[i].lower())
        tokens = data[i].split(' ')
        #去除停用词
        tagged_sent = pos_tag([i for i in tokens if i and not i in stopwords.words('english')])

        #获取词性，并进行归一化
        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        data[i] = lemmas_sent
    return data 


#求背景概率
def base_prob(labels):
    pos, neg = 0.0, 0.0
    for i in labels:
        if i == 'ham':
            neg += 1
        else:
            pos += 1
    return pos / (pos + neg), neg / (pos + neg)


#计算每本单词是否属于垃圾邮件的概率
def word_prob(data, labels):
    word_dict = {}
    for i in range(len(data)):
        lab = labels[i]
        dat = list(set(data[i]))

        for word in dat:
            if word not in word_dict:
                word_dict[word] = {'ham' : 1, 'spam' : 1}
            word_dict[word][lab] += 1

    for i in word_dict:
        dt = word_dict[i]
        ham = dt['ham']
        spam = dt['spam']
        word_dict[i]['ham'] = ham / float(ham + spam)
        word_dict[i]['spam'] = spam / float(ham + spam)
    return word_dict


def predict(samples, word_prob, base_p, base_n):
    ret = []
    for sam in samples:
        neg = math.log(base_n)
        pos = math.log(base_p)
        for word in sam:
            if word not in word_prob:
                continue
            neg += math.log(word_prob[word]['spam'])
            pos += math.log(word_prob[word]['ham'])
        ret.append('ham' if pos > neg else 'spam')
    return ret



labels, data = split_data(read_file("SMSSpamCollection.txt"))
data = data_profile(data)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25)

#计算训练集背景概率
base_p, base_n = base_prob(y_train)
word_dict = word_prob(x_train, y_train)
ret = predict(x_test, word_dict, base_p, base_n)
print(classification_report(y_test, ret))