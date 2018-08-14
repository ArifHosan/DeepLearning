import re
import string
from numpy import array
from pickle import dump
from unicodedata import normalize
from pickle import load
from numpy.random import shuffle


# Loads a file and returns contents as string.
# encoding used UTF-8
# TODO: check encoding for bengali language
def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


# saves file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# loads dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# takes string and returns tab separated pairs for each line
# TODO: add parameter so that lines are split using that delimeter
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


# takes lines and removes non printable characters and returns array
# TODO: check if this can clean bengali sentences or not
def clean_pairs(lines):
    cleaned = list()
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            line = line.split()
            line = [word.lower() for word in line]
            line = [word.translate(table) for word in line]
            line = [re_print.sub('', w) for w in line]
            line = [word for word in line if word.isalpha()]
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


# cleans raw data and saves in a single file.
# use spot_check value to check data pairs
def init_clean(spot_check=-1, src="deu-eng/deu.txt", dest="english-german"):
    print("Started Loading Raw Data File...................")

    filename = src
    doc = load_doc(filename)
    pairs = to_pairs(doc)
    clean_pairs_list = clean_pairs(pairs)

    print("Data File ", filename, " Loaded & Cleaned...................")

    dest_filename = dest + ".pkl"

    save_clean_data(clean_pairs_list, dest_filename)

    print("Cleaned File Saved as ", dest_filename)

    if spot_check > 0:
        print("Check Clean Files.................")
        for i in range(min(spot_check,1000)):
            print('[%s] => [%s]' % (clean_pairs_list[i, 0], clean_pairs_list[i, 1]))


# splits cleaned data into three files.
# suffix: both contains all the datasets.
# suffix: train contains data_size amounts of data for train purpose
# suffix: test contains rest of the data for test purpose
# change upper_limit to set max data size
def split_data(dest="english-german", data_size=10000):
    upper_limit = 100000
    dest_filename = dest + ".pkl"
    raw_dataset = load_clean_sentences(dest_filename)
    data_size = min(data_size, upper_limit)
    # dataset = raw_dataset[:data_size, :]
    dataset = raw_dataset
    shuffle(dataset)
    train, test = dataset[:10000], dataset[2000:]
    save_clean_data(dataset, dest + '-both.pkl')
    save_clean_data(train, dest + '-train.pkl')
    save_clean_data(test, dest + '-test.pkl')