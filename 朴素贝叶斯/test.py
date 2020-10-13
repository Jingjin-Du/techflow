import re

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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
        data[i] = re.sub(rule, '', data[i].lower())
        tokens = data[i].split(' ')
        tagged_sent = pos_tag([i for i in tokens if i and not i in stopwords.words('english')])

        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        data[i] = lemmas_sent
    return data 


labels, data = split_data(read_file("SMSSpamCollection.txt"))
data = data_profile(data)

print(data[:3])