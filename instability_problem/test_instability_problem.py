"""
Testing Framework for Decision-Focused Learning on the Instability Problem

This script evaluates Decision-Focused Learning (DFL) methods on a small
synthetic "instability" problem: a 2D linear objective is maximized over the
vertices of a triangle whose shape is controlled by a scale parameter b. As b
grows, the problem becomes increasingly ill-conditioned/unstable, which makes
it a useful stress test for DFL methods and their cost-regularized variants.

Configuration:
    Model parameters are loaded from a JSON file (see --config), enabling
    systematic evaluation of different approaches and hyperparameters.

Arguments:
    Problem Configuration:
    --instance (str): Problem class, controlling the scale parameter b, options:
                     1: b = 1       (b = 10**(instance-1))
                     2: b = 25      (explicit override -- see
                                     Trainer/instability_problem.py's
                                     INSTANCE_B_OVERRIDES)
                     3: b = 100     (b = 10**(instance-1))

    Model Configuration:
    --model (str): DFL model to evaluate (e.g., 'SPO', 'DBB', 'DPO')
    --loss (str): Loss function for training
    --config (str): Path to the JSON file listing the runs to execute:
                     'config.json' (default) runs the best/tuned
                     hyperparameters per model and instance; 'config_grid.json'
                     runs the full hyperparameter grid searched in the paper

    Training Parameters:
    --lr (float): Learning rate (default: 1e-3)
    --batch_size (int): Batch size (default: 128)
    --max_epochs (int): Maximum training epochs (default: 30)
    --l1_weight (float): L1 regularization weight (default: 1e-5)

    Model-Specific Parameters:
    --lambda_val (float): Interpolation parameter for blackbox differentiation (default: 1.0)
    --sigma (float): Noise parameter for DPO/FY methods (default: 1.0)
    --num_samples (int): Number of samples for FY (default: 1)
    --temperature (float): Temperature parameter for noise (default: 1.0)
    --nb_iterations (int): Number of iterations (default: 1)
    --k (int): Parameter k for specific methods (default: 10)
    --nb_samples (int): Number of samples parameter (default: 1)

Outputs (written under Rslt/, one row per seed unless noted):
    {model}ip{loss}{instance}.csv           -- aggregate test/val regret & mse
    {model}sampleOutput{loss}{instance}.csv    -- per-test-sample end-of-training
                                                   solution vertex (100 rows/seed)
    {model}valSampleOutput{loss}{instance}.csv -- the same, for the validation set
    regret_loss_tracker_{model}_{instance}.json -- per-epoch train/val loss &
        regret, plus (per seed) train_cost_norm_avg/max/min -- the average,
        max, and min L2 norm of the *raw* predicted cost vector y_hat across
        that seed's whole training run (before any of a model's own internal
        normalization), from log_cost_norm() in Trainer/PO_models.py
"""

import argparse
from argparse import Namespace
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import numpy as np
import torch
import shutil
import random
from Trainer.PO_models import *
from pytorch_lightning.callbacks import ModelCheckpoint
from Trainer.data_utils import DataModule
from Trainer.instability_problem import InstabilityProblem
from distutils.util import strtobool
import json

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
import copy
import matplotlib.pyplot as plt
import time
import sys

parser = argparse.ArgumentParser(description="Testing framework for Decision-Focused Learning on the instability problem")

# Problem configuration
parser.add_argument("--model", type=str, help="Name of the DFL model to evaluate (e.g., 'SPO', 'DBB', 'DPO')", default="", required=False)
parser.add_argument("--instance", type=str, help="Problem class: 1 and 3 give scale b = 10**(instance-1) = 1/100; 2 gives b = 25 (see INSTANCE_B_OVERRIDES in Trainer/instability_problem.py)", default="1", required=False)
parser.add_argument("--loss", type=str, help="Loss function for training", default="", required=False)
parser.add_argument("--config", type=str, help="Path to the JSON file listing the runs to execute: 'config.json' (default) runs the best/tuned hyperparameters per model and instance; 'config_grid.json' runs the full hyperparameter grid searched in the paper", default="config.json", required=False)

# Training parameters
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="Batch size", default=128, required=False)
parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs", default=30, required=False)
parser.add_argument("--l1_weight", type=float, help="Weight of L1 regularization", default=1e-5, required=False)

# Model-specific parameters
parser.add_argument("--lambda_val", type=float, help="Interpolation parameter for blackbox differentiation", default=1., required=False)
parser.add_argument("--sigma", type=float, help="DPO FY noise parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples FY", default= 1, required=False)

parser.add_argument("--temperature", type=float, help="input and target noise temperature parameter", default= 1., required=False)
parser.add_argument("--nb_iterations", type=int, help="number of iterations", default= 1, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples paprameter", default= 1, required=False)
parser.add_argument("--beta", type=float, help="parameter lambda of IMLE", default= 10., required=False)

parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default= 10., required=False)
parser.add_argument("--regularizer", type=str, help="Types of Regularization", default= 'quadratic', required=False)
parser.add_argument("--thr", type=float, help="threshold parameter", default= 1e-6)
parser.add_argument("--damping", type=float, help="damping parameter", default= 1e-8)
parser.add_argument("--diffKKT",  action='store_true', help="Whether KKT or HSD ",  required=False)

parser.add_argument("--tau", type=float, help="parameter of rankwise losses", default= 1e-8)
parser.add_argument("--growth", type=float, help="growth parameter of rankwise losses", default= 0.05)

parser.add_argument('--scheduler', dest='scheduler',  type=lambda x: bool(strtobool(x)), default= False)

class MetricTracker(Callback):

    def __init__(self):
        self.collection = []
        self.collection_test = []


    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        return

    def on_train_epoch_end(self, trainer, pl_module):
        elogs = trainer.logged_metrics
        self.collection.append(copy.deepcopy(elogs))
        return

    # def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, idk):
        # return

    # def on_validation_epoch_end(self, trainer, module):
        # elogs = trainer.logged_metrics
        # self.collection.append(copy.deepcopy(elogs))
        # return

    # def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx, idk):
        # return


    # def on_test_epoch_end(self, trainer, module):
        # elogs = trainer.logged_metrics
        # self.collection_test.append(copy.deepcopy(elogs))
        # return

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


def append_csv_matching_header(df, path):
    """Append df to path as CSV, keeping every appended row aligned with the
    file's existing column layout no matter what.

    Why this exists: which columns end up in `explicit` (and therefore in
    df) depends on which CLI flags/JSON keys happened to be given for THIS
    particular invocation -- e.g. a bare `python test_instability_problem.py`
    (relying on --config's default) never puts 'config' on the command line,
    so that run's rows would otherwise come out with fewer columns than a
    run that was launched with an explicit `--config ...`. Appending
    differently-shaped rows under one fixed header silently shifts every
    field after the point of divergence out of alignment -- that is
    precisely the corruption found in Rslt/*sampleOutput*.csv after a
    config.json run following earlier config_grid.json runs.

    The fix: if the file already exists, read its header and reindex df to
    that exact column list before appending, regardless of df's own column
    order or set. A column the header expects but df lacks is filled with
    NaN; a column df has that the header doesn't is dropped (and reported)
    rather than silently shifting everything else."""
    import os as _os
    import csv as _csv
    file_exists = _os.path.exists(path) and _os.path.getsize(path) > 0
    if file_exists:
        with open(path, "r", newline="") as f:
            header = next(_csv.reader(f))
        # to_csv (below, default index=True) always writes the index first,
        # so the header's first field is the index label ('instance'), not a
        # real data column -- only header[1:] corresponds to df's own
        # columns. (Every sampleOutputFile/valSampleOutputFile additionally
        # has a genuine 'instance' *data* column too, so 'instance' appears
        # twice in the header text; that's expected and handled correctly by
        # only ever reindexing against header[1:].)
        target_cols = header[1:]
        extra = [c for c in df.columns if c not in target_cols]
        if extra:
            print("WARNING: {} -- dropping column(s) {} not present in the file's "
                  "existing header so the append stays aligned. Delete the file "
                  "and rerun if these columns need to be captured.".format(path, extra))
        df = df.reindex(columns=target_cols)
    with open(path, "a", newline="") as f:
        df.to_csv(f, header=not file_exists)


def exec():

    config_args, _ = parser.parse_known_args()
    with open(config_args.config, "r") as json_file:
        parameter_sets = json.load(json_file)

    import os
    os.makedirs("Rslt", exist_ok=True)

    regret_loss_tracker=dict()
    cpt=0
    for parameters in parameter_sets:
        regret_loss_tracker[str(parameters)]=dict()

        Args = argparse.Namespace(**parameters)
        args = parser.parse_args(namespace=Args)
        argument_dict = vars(args)

        sentinel = _Sentinel()

        explicit_keys = {key: sentinel if key not in parameters else parameters[key] for key in argument_dict}
        sentinel_ns = Namespace(**explicit_keys)
        parser.parse_args(namespace=sentinel_ns)
        explicit = {key:value for key, value in vars(sentinel_ns).items() if value is not sentinel }
        print ("EXPLICIT",  explicit)

        get_class = lambda x: globals()[x]
        modelcls = get_class(argument_dict['model'])
        modelname = argument_dict.pop('model')
        instance = args.instance

        ######## Solver for this instance
        solver = InstabilityProblem(instance)
        # ###################################### Hyperparams #########################################

        torch.use_deterministic_algorithms(True)


        # ################## Define the outputfile
        outputfile = "Rslt/{}ip{}{}.csv".format(modelname, args.loss,  args.instance)
        sampleOutputFile = "Rslt/{}sampleOutput{}{}.csv".format(modelname,   args.loss,args.instance)
        valSampleOutputFile = "Rslt/{}valSampleOutput{}{}.csv".format(modelname,   args.loss,args.instance)
        ckpt_dir =  "ckpt_dir/{}{}{}/".format(modelname,  args.loss,args.instance)
        log_dir = "lightning_logs/{}{}{}/".format(modelname,  args.loss,args.instance)

        learning_curve_datafile = "LearningCurve/{}_".format(modelname)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items()] )+".csv"

        shutil.rmtree(log_dir,ignore_errors=True)

        for seed in range(10):
            regret_loss_tracker[str(parameters)][seed]=dict()

            # Train baseline mse
            torch.use_deterministic_algorithms(True)

            g = torch.Generator()
            g.manual_seed(seed)

            shutil.rmtree(ckpt_dir,ignore_errors=True)
            checkpoint_callback = ModelCheckpoint(
                            # monitor="val_regret",mode="min",
                            dirpath=ckpt_dir,
                            filename="model-{epoch:02d}-{val_regret:.8f}",

                        )
            seed_all(seed)

            data =  DataModule(generator=g, num_workers=0, solver=solver)
            tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
            callback_list = []
            cb = MetricTracker()
            callback_list.append(cb)
            callback_list.append(checkpoint_callback)
            # cb_stop = EarlyStopping(monitor="val_regret", mode="min", patience=1, min_delta=0.005)
            # callback_list.append(cb_stop)
            model = modelcls(solver=solver,seed=seed, **argument_dict)

            trainer = pl.Trainer(max_epochs=  argument_dict['max_epochs'], min_epochs=3,
            logger=tb_logger, callbacks=callback_list, check_val_every_n_epoch=1)

            t_start = time.process_time()
            trainer.fit(model, datamodule=data)
            training_time = time.process_time() - t_start

            # torch.save(model.state_dict(), "PF_model/model_"+str(argument_dict['instance'])+"_"+str(seed)+".pt")

            best_model_path = checkpoint_callback.best_model_path
            # print("Model Path:",best_model_path)
            if modelname=="CachingPO":
                model = modelcls.load_from_checkpoint(best_model_path ,  init_cache=cache, solver=solver,seed=seed,
            **argument_dict)
            else:
                model = modelcls.load_from_checkpoint(best_model_path ,solver=solver,seed=seed,
            **argument_dict)

            # Calculate and save output values (test set)
            output_list = trainer.predict(model, data.test_dataloader())

            df = pd.DataFrame({"output":output_list[0].tolist()})
            df.index.name='instance'
            for k,v in explicit.items():
                df[k] = v
            # Always stamp these explicitly rather than relying solely on
            # `explicit` -- config.json (as opposed to config_grid.json) was
            # never listed on this run's command line, so 'config' would
            # otherwise be silently missing from these columns. See
            # append_csv_matching_header()'s docstring for why that matters.
            df['config'] = config_args.config
            df['scheduler'] = bool(argument_dict.get('scheduler', False))
            df['seed'] = seed
            append_csv_matching_header(df, sampleOutputFile)

            # Calculate and save output values (validation set) -- mirrors the
            # test-set block above exactly, just on the validation dataloader.
            # This is what lets best_results.py-style analyses (and the
            # article's vertex-proportion figure) compare the end-of-training
            # per-sample solution choice on val vs test, not just aggregate
            # val_regret/val_mse.
            val_output_list = trainer.predict(model, data.val_dataloader())

            df = pd.DataFrame({"output":val_output_list[0].tolist()})
            df.index.name='instance'
            for k,v in explicit.items():
                df[k] = v
            df['config'] = config_args.config
            df['scheduler'] = bool(argument_dict.get('scheduler', False))
            df['seed'] = seed
            append_csv_matching_header(df, valSampleOutputFile)

            # Calculate and save performance
            validresult = trainer.validate(model,datamodule=data)
            testresult = trainer.test(model, datamodule=data)
            df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
            for k,v in explicit.items():
                df[k] = v
            df['config'] = config_args.config
            df['scheduler'] = bool(argument_dict.get('scheduler', False))
            df['seed'] = seed
            append_csv_matching_header(df, outputfile)

            clct = cb.collection

            val_regret = []
            train_regret = []
            val_loss = []
            train_loss = []
            # Per-epoch train_cost_norm_mean/max/min come from log_cost_norm()
            # in PO_models.py (logged on every training batch, PL reduces
            # each to one mean/max/min value per epoch already) -- collected
            # here per-epoch, then reduced once more below across the whole
            # run's epochs (mean-of-means, max-of-maxes, min-of-mins; exact
            # since every epoch has the same number of training batches).
            cost_norm_mean_per_epoch = []
            cost_norm_max_per_epoch = []
            cost_norm_min_per_epoch = []
            for i, metrics in enumerate(cb.collection, 1):
                train_regret.append(float(metrics['train_regret']))
                val_regret.append(float(metrics['val_regret']))
                train_loss.append(float(metrics['train_loss']))
                val_loss.append(float(metrics['val_loss']))
                if 'train_cost_norm_mean' in metrics:
                    cost_norm_mean_per_epoch.append(float(metrics['train_cost_norm_mean']))
                    cost_norm_max_per_epoch.append(float(metrics['train_cost_norm_max']))
                    cost_norm_min_per_epoch.append(float(metrics['train_cost_norm_min']))
            regret_loss_tracker[str(parameters)][seed]['val_regret']=val_regret
            regret_loss_tracker[str(parameters)][seed]['train_regret']=train_regret
            regret_loss_tracker[str(parameters)][seed]['val_loss']=val_loss
            regret_loss_tracker[str(parameters)][seed]['train_loss']=train_loss
            if cost_norm_mean_per_epoch:
                regret_loss_tracker[str(parameters)][seed]['train_cost_norm_avg'] = sum(cost_norm_mean_per_epoch) / len(cost_norm_mean_per_epoch)
                regret_loss_tracker[str(parameters)][seed]['train_cost_norm_max'] = max(cost_norm_max_per_epoch)
                regret_loss_tracker[str(parameters)][seed]['train_cost_norm_min'] = min(cost_norm_min_per_epoch)

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

        with open('Rslt/regret_loss_tracker_'+modelname+'_'+str(instance)+'.json', 'w') as f:
            json.dump(regret_loss_tracker, f)


        # df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
        # "val_mse": mses })
        # df['model'] = modelname
        # df.to_csv(learning_curve_datafile)


if __name__ == '__main__':
    exec()
