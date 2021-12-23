'''
This piece of code is used to initiate the training process of the model
'''
import time
from TransformerModel.HyperParameters import HyperParameters
from TransformerModel.Model import get_model
from TransformerModel.Preprocessor import *
import torch.nn.functional as F
from TransformerModel.Optimizer import CosineWithRestarts
from TransformerModel.Batching import create_masks
import dill as pickle
# Import Useful Libraries
import shutil as sh
i

'''
Train_model() function is used to train the model for the requested number of epochs.
1)model contain the parameters and layers of model
2)Hparameters contain HyperParameters like batchsize,learning rate etc of model
'''


def train_model(model, HParameters):
    print("model is training...")
    start = time.time()
    if HParameters.checkpoint > 0:
        cptime = time.time()
    # loop for number of epochs
    for epoch in range(HParameters.epochs):
        model.train()
        total_loss = 0

        if HParameters.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
        # loop for number of iterations
        for i, batch in enumerate(HParameters.train):
            if HParameters.device == "cuda":
                src = batch.src.transpose(0, 1).cuda()
                trg = batch.trg.transpose(0, 1).cuda()
            else:
                src = batch.src.transpose(0, 1)
                trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, HParameters)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            HParameters.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=HParameters.trg_pad)
            loss.backward()
            HParameters.optimizer.step()

            HParameters.sched.step()
            total_loss += loss.item()
            # If checkpoint is requested then keep saving the model weights after every checkpoint minutes
            if HParameters.checkpoint > 0 and ((time.time() - cptime) // 60) // HParameters.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()
            # Print The loss
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
            ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)),
            "".join(' ' * (20 - (100 // 5))), 100,
            total_loss / get_len(HParameters.train)))

        print("model is validating...")
        model.eval()
        val_loss=0
        start = time.time()
        # loop for number of epochs
        with torch.no_grad():
            for i, batch in enumerate(HParameters.val):
                if HParameters.device == "cuda":
                    src = batch.src.transpose(0, 1).cuda()
                    trg = batch.trg.transpose(0, 1).cuda()
                else:
                    src = batch.src.transpose(0, 1)
                    trg = batch.trg.transpose(0, 1)
                trg_input = trg[:, :-1]
                src_mask, trg_mask = create_masks(src, trg_input, HParameters)
                preds = model(src, trg_input, src_mask, trg_mask)
                ys = trg[:, 1:].contiguous().view(-1)
                HParameters.optimizer.zero_grad()
                loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=HParameters.trg_pad)

                val_loss += loss.item()
                # If checkpoint is requested then keep saving the model weights after every checkpoint minutes
                if HParameters.checkpoint > 0 and ((time.time() - cptime) // 60) // HParameters.checkpoint >= 1:
                    torch.save(model.state_dict(), 'weights/model_weights')
                    cptime = time.time()
                # Print The loss
            print("%dm: val epoch %d [%s%s]  %d%% val loss = %.3f\nepoch %d complete, loss = %.03f val loss = %.03f" % \
                ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)),
                "".join(' ' * (20 - (100 // 5))), 100,
                val_loss / get_len(HParameters.val), epoch + 1,total_loss / get_len(HParameters.train), val_loss / get_len(HParameters.val)))


# main function that is used to initiate the training process
def main():
    # set Hyperparameters for the model
    Hparameters = HyperParameters(src_data="data/Urd2T.txt", trg_data="data/Eng2T.txt",
                        src_lang="ur", trg_lang="en",batchsize=1200,epochs=2,checkpoint=20,
                        lr=0.0001,dropout=0.4,d_model=512,load_weights="weights")
    # Use cuda GPU if available otherwise Use cpu
    Hparameters.device = "cuda" if torch.cuda.is_available() else "cpu"
    read_files(Hparameters)
    # Create_Fields for both the input languages
    Source, Target = make_fields(Hparameters)
    # Make dataset using above fields
    Hparameters.train, Hparameters.val = make_dataset(Hparameters, Source, Target)
    model = get_model(Hparameters, len(Source.vocab), len(Target.vocab))
    # set Optimizer for the model
    Hparameters.optimizer = torch.optim.Adam(model.parameters(), lr=Hparameters.lr, betas=(0.9, 0.98), eps=1e-9)
    Hparameters.sched = CosineWithRestarts(Hparameters.optimizer, T_max=Hparameters.train_len)
    train_model(model, Hparameters)
    #set the name for the folder where fields and weights will be stored
    destination = "weights"
    #if Hparameters.load_weights is not None:
     #   sh.rmtree(Hparameters.load_weights)
    #if Hparameters.checkpoint > 0:
    #    sh.rmtree(Hparameters.load_weights)
    #Make a folder with name destination
    #os.makedirs(destination)
    print("saving weights to " + destination + "/...")
    #Save the trained weights of the model
    torch.save(model.state_dict(), destination+'/model_weights')
    #save Both the source and target Fields
    pickle.dump(Source, open(destination+'/Source.pkl', 'wb'))
    pickle.dump(Target, open(destination+'/Target.pkl', 'wb'))
    print("Field pickles and weights saved to " + destination)

if __name__ == "__main__":
    main()
