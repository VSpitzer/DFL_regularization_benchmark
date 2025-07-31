import argparse
from argparse import Namespace
import pytorch_lightning as pl
import shutil
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Trainer.PO_models import *
from Trainer.data_utils import KnapsackDataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback
from distutils.util import strtobool
import json
import copy
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument("--capacity", type=int, help="capacity of knapsack", default= 12, required=False)
parser.add_argument("--model", type=str, help="name of the model", default= "", required= False)
parser.add_argument("--loss", type= str, help="loss", default= "", required=False)

parser.add_argument("--lambda_val", type=float, help="interpolaton parameter blackbox", default= 1., required=False)
parser.add_argument("--sigma", type=float, help="DPO FY noise parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples FY", default= 1, required=False)

parser.add_argument("--temperature", type=float, help="input and target noise temperature parameter", default= 1., required=False)
parser.add_argument("--nb_iterations", type=int, help="number of iterations", default= 1, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples paprameter", default= 1, required=False)
parser.add_argument("--beta", type=float, help="parameter lambda of IMLE", default= 10., required=False)


parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default= 10., required=False)
parser.add_argument("--thr", type=float, help="threshold parameter", default= 1e-6)
parser.add_argument("--damping", type=float, help="damping parameter", default= 1e-8)
parser.add_argument("--tau", type=float, help="parameter of rankwise losses", default= 1e-8)
parser.add_argument("--growth", type=float, help="growth parameter of rankwise losses", default= 0.05)

parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum number of epochs", default= 35, required=False)
parser.add_argument("--num_workers", type=int, help="maximum number of workers", default= 4, required=False)
parser.add_argument('--scheduler', dest='scheduler',  type=lambda x: bool(strtobool(x)))

class MetricTracker(Callback):

  def __init__(self):
    self.collection = []
    self.collection_test = []

  def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, idk):
    # vacc = outputs['val_acc'] # you can access them here
    # self.collection.append(vacc) # track them
    return

  def on_validation_epoch_end(self, trainer, module):
    elogs = trainer.logged_metrics # access it here
    self.collection.append(copy.deepcopy(elogs))
    # do whatever is needed
    return 
    
  def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx, idk):
    # vacc = outputs['val_acc'] # you can access them here
    # self.collection.append(vacc) # track them
    return

    
  def on_test_epoch_end(self, trainer, module):
    elogs = trainer.logged_metrics # access it here
    self.collection_test.append(copy.deepcopy(elogs))
    # do whatever is needed
    return 
    
class _Sentinel:
    pass
sentinel = _Sentinel()
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def exec():
    # Load parameter sets from JSON filw
    with open('config_DBB.json', "r") as json_file:
        parameter_sets = json.load(json_file)

    cpt=0

    for parameters in parameter_sets:
        
        Args = argparse.Namespace(**parameters)
        args = parser.parse_args(namespace=Args)
        argument_dict = vars(args)
        # print(argument_dict)
        
        sentinel = _Sentinel()
        
        explicit_keys = {key: sentinel if key not in parameters else parameters[key] for key in argument_dict}
        sentinel_ns = Namespace(**explicit_keys)
        parser.parse_args(namespace=sentinel_ns)
        explicit = {key:value for key, value in vars(sentinel_ns).items() if value is not sentinel }
        print ("EXPLICIT",  explicit)
        get_class = lambda x: globals()[x]
        modelcls = get_class(argument_dict['model'])
        modelname = argument_dict.pop('model')
        capacity= argument_dict.pop('capacity')
        torch.use_deterministic_algorithms(True)

        ################## Define the outputfile
        outputfile = "Rslt/{}Knapsack{}.csv".format(modelname,args.loss)
        regretfile = "Rslt/{}KnapsackRegret{}.csv".format( modelname,args.loss )
        ckpt_dir =  "ckpt_dir/{}{}/".format( modelname,args.loss)
        log_dir = "lightning_logs/{}{}/".format(  modelname,args.loss )
        learning_curve_datafile = "LearningCurve/{}_".format(modelname)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items()] )+".csv"  

        shutil.rmtree(log_dir,ignore_errors=True)

        for seed in range(10):
            seed_all(seed)
            shutil.rmtree(ckpt_dir,ignore_errors=True)
            checkpoint_callback = ModelCheckpoint(
                    #monitor="val_regret",mode="min",
                    dirpath=ckpt_dir, 
                    filename="model-{epoch:02d}-{val_regret:.8f}",
                    )

            g = torch.Generator()
            g.manual_seed(seed)    
            data =  KnapsackDataModule(capacity=  capacity, batch_size= argument_dict['batch_size'], generator=g, seed= seed, num_workers= 4)
            weights, n_items =  data.weights, data.n_items
            if modelname=="CachingPO":
                cache = torch.from_numpy (data.train_df.sol)
            tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
            
            callback_list = []
            cb = MetricTracker()
            callback_list.append(cb)
            callback_list.append(checkpoint_callback)
            # cb_stop = EarlyStopping(monitor="val_regret", mode="min", patience=2, min_delta=0.005)
            # callback_list.append(cb_stop)
            trainer = pl.Trainer(max_epochs= argument_dict['max_epochs'], log_every_n_steps=1, 
            min_epochs=1,logger=tb_logger, callbacks=callback_list, check_val_every_n_epoch=10)
            if modelname=="CachingPO":
                model =  modelcls(weights,capacity,n_items,init_cache=cache,seed=seed, **argument_dict)
            else:
                model =  modelcls(weights,capacity,n_items,seed=seed, **argument_dict)
            
            # model.load_state_dict(torch.load("models/baseline_mse_"+str(capacity)+"_"+str(seed)+".pt"))

            t_start = time.process_time()
            trainer.fit(model, datamodule=data)
            training_time = time.process_time() - t_start
            best_model_path = checkpoint_callback.best_model_path

            if modelname=="CachingPO":
                model =  modelcls.load_from_checkpoint(best_model_path,
                weights = weights,capacity= capacity,n_items = n_items,init_cache=cache,seed=seed, **argument_dict)
            else:
                model =  modelcls.load_from_checkpoint(best_model_path,
                weights = weights,capacity= capacity,n_items = n_items,seed=seed, **argument_dict)
                
            # torch.save(model.state_dict(), "models/baseline_mse_"+str(capacity)+"_"+str(seed)+".pt")

        ##### SummaryWrite ######################
            validresult = trainer.validate(model,datamodule=data)
            testresult = trainer.test(model, datamodule=data)
            df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
            for k,v in explicit.items():
                df[k] = v
            df['seed'] = seed
            df['capacity'] =capacity
            df['time'] =training_time
            with open(outputfile, 'a') as f:
                    df.to_csv(f, header=f.tell()==0)


            regret_list = trainer.predict(model, data.test_dataloader())
            

            df = pd.DataFrame({"regret":regret_list[0].tolist()})
            df.index.name='instance'
            for k,v in explicit.items():
                df[k] = v
            df['seed'] = seed
            df['capacity'] =capacity    
            with open(regretfile, 'a') as f:
                df.to_csv(f, header=f.tell()==0)
                
            clct = cb.collection
            # var_tab = [e['avg_var_step'] for e in clct[1:-1]]
            # min_var_tab = [e['min_var_step'] for e in clct[1:-1]]
            # max_var_tab = [e['max_var_step'] for e in clct[1:-1]]
            
            # val_regret_tab = [e['val_regret'] for e in clct[1:-1]]
            # loss_tab = [e['train_loss_step'] for e in clct[1:-1]]
        
            # figurefile = "C:/Users/Victor/Pictures/ADPO/"
            
            # plt.plot(np.arange(len(var_tab)), var_tab)
            # plt.title("Var")
            # plt.savefig(figurefile+str(seed)+"_Var_ADPO.png")
            # plt.close()
            
            
            # plt.plot(np.arange(len(loss_tab)), loss_tab)
            # plt.title("Train loss")
            # plt.savefig(figurefile+str(seed)+"_Loss.png")
            # plt.close()
            
            # plt.plot(np.arange(len(val_regret_tab)), val_regret_tab)
            # plt.title("Train regret")
            # plt.savefig(figurefile+str(seed)+"_Regret.png")
            # plt.close()

            cpt+=1



        ###############################  Save  Learning Curve Data ########
        import os
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        parent_dir=   log_dir+"lightning_logs/"
        version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

        walltimes = []
        steps = []
        regrets= []
        mses = []
        for logs in version_dirs:
            event_accumulator = EventAccumulator(logs)
            event_accumulator.Reload()

            events = event_accumulator.Scalars("val_regret")
            walltimes.extend( [x.wall_time for x in events])
            steps.extend([x.step for x in events])
            regrets.extend([x.value for x in events])
            events = event_accumulator.Scalars("val_mse")
            mses.extend([x.value for x in events])

        df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
        "val_mse": mses })
        df['model'] = modelname
        df.to_csv(learning_curve_datafile)


if __name__ == '__main__':
    exec()

            
