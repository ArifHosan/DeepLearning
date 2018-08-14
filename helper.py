from cleaner import load_clean_sentences


def load_datasets(base_filename="english-german"):
    dataset = load_clean_sentences(base_filename + '-both.pkl')
    train = load_clean_sentences(base_filename + '-train.pkl')
    test = load_clean_sentences(base_filename + '-test.pkl')
    return dataset, train, test

