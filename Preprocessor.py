# import the rquired libraries
import pandas as pd
import os
import dill as pickle
import spacy
import torch

from torchtext.legacy import data
from Batching import MyIterator, batch_size_function
torch.cuda.empty_cache()

# find and return the length of the train data
def get_len(train):
    return sum(1 for _ in train)


#Function that is used to load data from text file to numpy array
def read_files(Hparameters):
    Hparameters.src_data = open(Hparameters.src_data, encoding='utf8').read().split('\n')
    Hparameters.trg_data = open(Hparameters.trg_data, encoding='utf8').read().split('\n')

#Make the datasets of the fields
def make_dataset(Hparameters, Source, Target,val_size=0.1):
    print("creating dataset and iterator... ")

    raw_data = {'src': [line for line in Hparameters.src_data], 'trg': [line for line in Hparameters.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    #train_df_=df.sample(frac=1-test_size,random_state=200) #random state is a seed value
    #test_df=df.drop(train_df_.index)
    train_df=df.sample(frac=1-val_size,random_state=200) #random state is a seed value
    val_df=df.drop(train_df.index)
    print(len(train_df.index),len(val_df.index))#,len(test_df.index))
    mask_train = (train_df['src'].str.count(' ') < Hparameters.max_strlen) & (train_df['trg'].str.count(' ') < Hparameters.max_strlen)
    mask_val = (val_df['src'].str.count(' ') < Hparameters.max_strlen) & (val_df['trg'].str.count(' ') < Hparameters.max_strlen)
    #mask_test = (test_df['src'].str.count(' ') < Hparameters.max_strlen) & (test_df['trg'].str.count(' ') < Hparameters.max_strlen)

    train_df = train_df.loc[mask_train]
    #test_df = test_df.loc[mask_test]
    val_df = val_df.loc[mask_val]

    train_df.to_csv("translate_transformer_temp_train.csv", index=False)
    val_df.to_csv("translate_transformer_temp_val.csv", index=False)
    #test_df.to_csv("translate_transformer_temp_test.csv", index=False)

    data_fields = [('src', Source), ('trg', Target)]
    train = data.TabularDataset('./translate_transformer_temp_train.csv', format='csv', fields=data_fields)
    valid = data.TabularDataset('./translate_transformer_temp_val.csv', format='csv', fields=data_fields)
    #test = data.TabularDataset('./translate_transformer_temp_test.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=Hparameters.batchsize, device=torch.device(Hparameters.device),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_function, train=True, shuffle=True)
    valid_iter = MyIterator(valid, batch_size=Hparameters.batchsize, device=torch.device(Hparameters.device),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_function, train=False, shuffle=True)
    #test_iter = MyIterator(test, batch_size=1, device=torch.device(Hparameters.device),
          #                  repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
           #                 batch_size_fn=batch_size_function, train=False, shuffle=True)
    os.remove('translate_transformer_temp_train.csv')
    os.remove('translate_transformer_temp_val.csv')

    if Hparameters.load_weights is None:
        Source.build_vocab(train)
        Target.build_vocab(train)
        if Hparameters.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(Source, open('weights/Source.pkl', 'wb'))
            pickle.dump(Target, open('weights/Target.pkl', 'wb'))

    Hparameters.src_pad = Source.vocab.stoi['<pad>']
    Hparameters.trg_pad = Target.vocab.stoi['<pad>']
   # print(get_len(train_iter))
    Hparameters.train_len = get_len(train_iter)

    return train_iter, valid_iter#, test_iter


def make_fields(Hparameters):
    spacy_langs = ['en', 'ur']
    if Hparameters.src_lang not in spacy_langs:
        print('invalid src language: ' + Hparameters.src_lang + 'supported languages : ' + spacy_langs)
    if Hparameters.trg_lang not in spacy_langs:
        print('invalid trg language: ' + Hparameters.trg_lang + 'supported languages : ' + spacy_langs)

    print("loading tokenizers...")

    t_src = spacy.blank(Hparameters.src_lang)
    t_trg = spacy.blank(Hparameters.trg_lang)

    def tokenize_src(text):
        return [tok.text for tok in t_src.tokenizer(text)]

    def tokenize_trg(text):
        return [tok.text for tok in t_trg.tokenizer(text)]

    Target = data.Field(lower=True, tokenize=tokenize_trg, init_token='<sos>', eos_token='<eos>')
    Source = data.Field(lower=True, tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>')
    if Hparameters.load_weights is not None:
        try:
            print("loading presaved fields...")
            Source = pickle.load(open(f'{Hparameters.load_weights}/Source.pkl', 'rb'))
            Target = pickle.load(open(f'{Hparameters.load_weights}/Target.pkl', 'rb'))
        except:
            print(
                "error opening Source.pkl and Target.pkl field files, please ensure they are in " + Hparameters.load_weights + "/")
            quit()

    return Source, Target
