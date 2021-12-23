'''Purpose of this piece of code is to convert the dataset into batchs
and make iterators for them.Also we have applied masking'''

# Import useful libraries
import numpy as np
from torch.autograd import Variable
import torch
from torchtext.legacy import data

# create two global variables for source and target data
global max_src_in_batch, max_tgt_in_batch

'''MyIterator class inherits the functionality of pytorch iterator.
Functionality of Myiterator is easy loading of dataset,organizing 
batching and masking'''


class MyIterator(data.Iterator):
    # create batches and organize them
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def nopeak_mask(size, HParameter):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if HParameter.device == "cuda":
        np_mask = np_mask.cuda()
    return np_mask


''' Create masks :It is a way to tell sequence-processing layers that certain timesteps
 in an input are missing, and thus should be skipped when processing the data'''


def create_masks(src, trg, Hparameter):
    src_mask = (src != Hparameter.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != Hparameter.trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, Hparameter)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


def batch_size_function(new, count,_):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
