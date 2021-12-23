'''This class is used to contain the information of all parameters used
in model training and translating sentences'''
class HyperParameters:
    def __init__(self, src_data, trg_data, src_lang, trg_lang, k=1,epochs=1, d_model=512,
                 n_layers=6, heads=8, dropout=0.1, batchsize=512, lr=0.0001
                 , load_weights=None, max_strlen=80, checkpoint=0):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.val = None
        self.test = None
        self.epochs = epochs
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        self.batchsize = batchsize
        self.k=k
        self.lr = lr
        self.load_weights = load_weights
        self.max_strlen = max_strlen
        self.checkpoint = checkpoint