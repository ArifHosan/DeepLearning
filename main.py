from cleaner import init_clean, split_data
from trainer import *
from validation import evaluate


def main():
    # load and split raw data in 3 different file for training purposes
    # init_clean(spot_check=20,src="data/deu.txt", dest="data/english-german")
    split_data(dest="data/english-german", data_size=10000)
    train_model(base_filename="data/english-german",model_filename="model/model")
    evaluate(filename="data/english-german", modelname="model/model")

if __name__ == "__main__":
    main()