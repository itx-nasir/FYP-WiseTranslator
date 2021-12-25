import nltk
from HyperParameters import HyperParameters
from BeamSearch import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
from Preprocessor import *
from Model import get_model
nltk.download('wordnet')

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]

    return 0


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opt, SRC, TRG):
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == "cuda":
        sentence = sentence.cuda()

    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)


def translate(opt, model, SRC, TRG):
    sentences=[]
    deli='.'
    if(opt.src_lang=='en'):
        sentences = opt.text.lower().split('.')
        
    elif(opt.src_lang=='ur'):
        sentences = opt.text.lower().split(chr(1748))
        deli=chr(1748)
        
    translated = []
    print(sentences)
    for sentence in sentences:
        if(not(sentence and sentence.strip())):
            pass
        else:
            translated.append(translate_sentence(sentence + deli, model, opt, SRC, TRG).capitalize())

    return ' '.join(translated)

def loadETUModels():
    print("Loading EnglishToUrdu Model...")
    global Hparameters1 
    Hparameters1= HyperParameters(src_data="data/Eng12.txt", trg_data="data/Urd12.txt",
                                  load_weights="weightsETU", k=3,max_strlen=100, d_model=512,
                                  n_layers=6, src_lang="en",trg_lang="ur", heads=8,
                                  dropout=0.3)

    Hparameters1.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert Hparameters1.k > 0
    assert Hparameters1.max_strlen > 10
    global SRC1
    global TRG1
    SRC1, TRG1 = make_fields(Hparameters1)
    global model1
    model1 = get_model(Hparameters1, len(SRC1.vocab), len(TRG1.vocab))

def loadUTEModels():
    print("Loading UrduTOEnglish Model...")
    global Hparameters2 
    Hparameters2= HyperParameters(src_data="data/UrdT2.txt", trg_data="data/EngT2.txt",
                                  load_weights="weightsUTE", k=3,max_strlen=100, d_model=512,
                                  n_layers=6, src_lang="ur",trg_lang="en", heads=8,
                                  dropout=0.3)

    Hparameters2.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert Hparameters2.k > 0
    assert Hparameters2.max_strlen > 10
    global SRC2
    global TRG2
    SRC2, TRG2 = make_fields(Hparameters2)
    global model2
    model2 = get_model(Hparameters2, len(SRC2.vocab), len(TRG2.vocab))

def Eng_to_Urd(text):
    Hparameters1.text=text
    phrase = translate(Hparameters1, model1, SRC1, TRG1)
    return phrase
def Urd_to_Eng(text):
    Hparameters2.text=text
    phrase = translate(Hparameters2, model2, SRC2, TRG2)
    return phrase
