from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from cleaner import load_clean_sentences
from helper import load_datasets
from model_definition import *


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# sequences the tokenized datasets
# and pads them for uniform lentgth
# padding is done after the sequence
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


def max_length(lines):
    return max(len(line.split()) for line in lines)


def get_tokenized(base_filename="english-german"):
    # loads all the datasets
    dataset, train, test = load_datasets(base_filename)
    # tokenize english datas
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % (eng_length))

    # tokenize german datas
    ger_tokenizer = create_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = max_length(dataset[:, 1])
    print('German Vocabulary Size: %d' % ger_vocab_size)
    print('German Max Length: %d' % (ger_length))

    return eng_tokenizer,ger_tokenizer



def train_model(base_filename="english-german", model_filename="model"):
    # loads all the datasets
    dataset, train, test = load_datasets(base_filename)
    # tokenize english datas
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % (eng_length))

    # tokenize german datas
    ger_tokenizer = create_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = max_length(dataset[:, 1])
    print('German Vocabulary Size: %d' % ger_vocab_size)
    print('German Max Length: %d' % (ger_length))

    # prepare datasets for train
    trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
    trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
    trainY = encode_output(trainY, eng_vocab_size)

    # prepare datasets for tests
    testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
    testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    testY = encode_output(testY, eng_vocab_size)

    # retrieves the model
    # current model is NMT
    # TODO: A New Model
    print("Compiling Model.......")
    model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    summarize_model(model,output_file=model_filename+ ".png")
    fit_model(model,trainX,trainY,testX,testY,filename=model_filename+".h5")


