import argparse
import pandas as pd
from HyperParameters import HyperParameters
from Preprocessor import *
from Translator import *
import nltk
import numpy as np
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
nltk.download('wordnet')

def evaluate_model(source_path, target_path):
    Hparameters = HyperParameters(src_data=source_path, trg_data=target_path,
                                  load_weights="weights", k=3,max_strlen=80, d_model=512,
                                  n_layers=6, src_lang="en",trg_lang="ur", heads=8,
                                  dropout=0.1)

    Hparameters.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert Hparameters.k > 0
    assert Hparameters.max_strlen > 10

    SRC, TRG = make_fields(Hparameters)
    model = get_model(Hparameters, len(SRC.vocab), len(TRG.vocab))
    original =[]
    translations =[]

    if target_path is None and source_path is None:
        df = pd.read_csv('translate_transformer_temp_test.csv')
        source = df['src'].values
        target = df['trg'].values
    else:
        src_df = pd.read_csv(source_path, delimiter='                   ', header=None)
        trg_df = pd.read_csv(target_path,delimiter='                   ', header=None)
        source = src_df.values.ravel().tolist()
        target = trg_df.values.ravel().tolist()

    scores=[]
    for src, trg in tqdm(zip(source,target)):
        Hparameters.text = src
        phrase = translate(Hparameters, model, SRC, TRG)
        original.append(trg)
        translations.append(phrase)
        cc = SmoothingFunction()
        try:
            score = sentence_bleu(phrase, trg, smoothing_function=cc.method4)
        except:
            score = bleu_score([phrase], [trg])

        print(Hparameters.text)
        print('> ' + phrase + '\n')
        print(score)
        
        scores.append(score)
    print('BLEU Score on Test Set: ', np.array(scores).mean())

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Machine Translation')
    parser.add_argument('--src', metavar='STR', type=str, default=None,
                        help='Source file')
    parser.add_argument('--trg', metavar='STR', type=str, default=None,
                        help='Target file')
    args = parser.parse_args()

    evaluate_model(args.src, args.trg)
