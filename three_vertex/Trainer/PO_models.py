import os
from Trainer.utils import regret_fn, regret_list, growpool_fn, abs_regret_fn
from Trainer.diff_layer import *
from DPO import perturbations
from DPO import fenchel_young as fy
import pandas as pd

import numpy as np 
from torch import nn, optim
from tqdm.auto import tqdm
import torch 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl


def log_cost_norm(module, y_hat):
    """Log the per-sample L2 norm of this training batch's *raw* predicted
    cost vector y_hat -- i.e. before any of a model's own internal
    normalization (the _rn/_rp variants divide y_hat by its norm, or by
    1 + norm/kappa, before ever using it), so this reflects what the network
    naturally produces. Logged per training step with on_epoch=True and a
    mean/max/min reduction, so PyTorch Lightning gives one
    mean/max/min-over-the-epoch's-batches value per epoch; test_instability_
    problem.py then further reduces those across the whole run (mean of the
    per-epoch means, max of the per-epoch maxes, min of the per-epoch mins)
    and best_results.py reports that, averaged across seeds, as the
    "cost vector norm throughout training" columns."""
    norms = y_hat.norm(dim=-1) if y_hat.dim() > 1 else y_hat.abs().unsqueeze(0)
    module.log("train_cost_norm_mean", norms.mean(), on_step=False, on_epoch=True, reduce_fx="mean")
    module.log("train_cost_norm_max", norms.max(), on_step=False, on_epoch=True, reduce_fx="max")
    module.log("train_cost_norm_min", norms.min(), on_step=False, on_epoch=True, reduce_fx="min")


class baseline_mse(pl.LightningModule):
    def __init__(self,solver,lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__()
        pl.seed_everything(seed)
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        
        self.lr = lr
        self.solver = solver
        self.scheduler = scheduler
        self.save_hyperparameters("lr")

    def forward(self,x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    def validation_step(self, batch, batch_idx):
        solver = self.solver
        
        x,y,sol,m = batch

        y_hat =  self(x).squeeze()
        val_loss= regret_fn(solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)

        self.log("val_regret", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
       
        return  {"val_regret": val_loss, "val_mse": mseloss}

    def test_step(self, batch, batch_idx):
        solver = self.solver
        
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)

        self.log("test_regret", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("test_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )    
        self.log("test_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, )        
        return  {"test_regret": val_loss, "test_mse": mseloss}
    def predict_step(self, batch, batch_idx):
        '''
        I am using the predict module to compute regret !
        '''
        solver = self.solver
        
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        per_sample_sol = torch.zeros(len(y_hat))
        for i in range(len(y_hat)):
            per_sample_sol[i] = self.solver.get_vertex_solve(y_hat[i].detach().numpy(),m[i].detach().numpy())
        return per_sample_sol

    def configure_optimizers(self):
        
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6),
                    "monitor": "val_regret"
            }
            }
        return optimizer


class SPO(baseline_mse):
    def __init__(self,solver,lr=1e-1,alpha=2,seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,seed,scheduler)
        self.layer = SPOlayer(solver, alpha=alpha)

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)
        loss =  self.layer(y_hat, y,sol,m )
        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret",train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        x,y,sol,m = batch

        y_hat =  self(x).squeeze()
        val_regret= regret_fn(self.solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(self.solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)

        val_loss =  self.layer(y_hat, y,sol,m )
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}

class SPO_rn(baseline_mse):
    def __init__(self,solver,lr=1e-1,alpha=2,seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,seed,scheduler)
        self.layer = SPOlayer(solver, alpha=alpha)

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])

        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.linalg.norm(y[i])


        loss = self.layer(normalized_y_hat, normalized_y, sol, m)

        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret",train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        x,y,sol,m = batch
    
        y_hat =  self(x).squeeze()
        val_regret= regret_fn(self.solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(self.solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])

        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.linalg.norm(y[i])
        val_loss =  self.layer(normalized_y_hat, normalized_y, sol, m)
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}
        
class SPO_rp(baseline_mse):
    def __init__(self,solver,lr=1e-1,alpha=2,kappa = 1, seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,seed,scheduler)
        self.layer = SPOlayer(solver, alpha=alpha)
        self.kappa = kappa

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./(1+torch.linalg.norm(y_hat[i])*1./self.kappa)

        loss = self.layer(normalized_y_hat, y, sol, m)

        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret",train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        x,y,sol,m = batch
    
        y_hat =  self(x).squeeze()
        val_regret= regret_fn(self.solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(self.solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./(1+torch.linalg.norm(y_hat[i])*1./self.kappa)   
        
        val_loss = self.layer(normalized_y_hat, y, sol, m)
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}

class DBB(baseline_mse):
    def __init__(self,solver,lr=1e-1,lambda_val=1.,seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,seed,scheduler)
        self.layer = DBBlayer(solver, lambda_val=lambda_val)

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)
        sol_hat  =  self.layer(y_hat, y,sol,m )
        loss = ((sol - sol_hat)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret",train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss
        
    def validation_step(self, batch, batch_idx):
        x,y,sol,m = batch
    
        y_hat =  self(x).squeeze()
        val_regret= regret_fn(self.solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(self.solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)
        
        sol_hat  =  self.layer(y_hat, y,sol,m )
        val_loss = ((sol - sol_hat)*y).sum(-1).mean()
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}


class DBB_rn(baseline_mse):
    def __init__(self,solver,lr=1e-1,lambda_val=1.,seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,seed,scheduler)
        self.layer = DBBlayer(solver, lambda_val=lambda_val)

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])

        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.linalg.norm(y[i])

        sol_hat  =  self.layer(normalized_y_hat, normalized_y,sol,m )

        loss = ((sol - sol_hat)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret",train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        x,y,sol,m = batch

        y_hat =  self(x).squeeze()
        val_regret= regret_fn(self.solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(self.solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])

        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.linalg.norm(y[i])

        sol_hat  =  self.layer(normalized_y_hat, normalized_y,sol,m )
        val_loss = ((sol - sol_hat)*y).sum(-1).mean()
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}

class DBB_rp(baseline_mse):
    def __init__(self,solver,lr=1e-1,lambda_val=1.,kappa=1, seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,seed,scheduler)
        self.layer = DBBlayer(solver, lambda_val=lambda_val)
        self.kappa = kappa

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./(1+torch.linalg.norm(y_hat[i])*1./self.kappa)

        sol_hat  =  self.layer(normalized_y_hat, y,sol,m )

        loss = ((sol - sol_hat)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret",train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss
        
    def validation_step(self, batch, batch_idx):
        x,y,sol,m = batch
    
        y_hat =  self(x).squeeze()
        val_regret= regret_fn(self.solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(self.solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./(1+torch.linalg.norm(y_hat[i])*1./self.kappa)  
                
        sol_hat  =  self.layer(normalized_y_hat, y,sol,m )
        val_loss = ((sol - sol_hat)*y).sum(-1).mean()
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}

class DPO(baseline_mse):
    def __init__(self,solver,instance=1,sigma=0.1,num_samples=10, lr=1e-1,seed=0,scheduler=False, logger_backend="tb", **kwd):
        self.sigma = sigma
        self.num_samples = num_samples
        self.instance = instance
        self.lr = lr
        self.logger_backend = logger_backend 
        self.seed = seed
        super().__init__(solver,lr,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)

        loss = 0
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma,noise='normal', batched= False)( y_hat[i] )
            loss += y[i].dot(sol[i] - op)
        loss /= len(y_hat)


        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret",train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss
        
    def validation_step(self, batch, batch_idx):
        x,y,sol,m = batch
    
        y_hat =  self(x).squeeze()
        val_regret = regret_fn(self.solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(self.solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)
       
        val_loss = 0
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma, noise='normal', batched= False)( y_hat[i] )
            val_loss += y[i].dot(sol[i] - op)
        val_loss /= len(y_hat)
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}
        
    
class DPO_rn(baseline_mse):
    def __init__(self,solver,instance=1,sigma=0.1,num_samples=10, lr=1e-1,seed=0,scheduler=False, logger_backend="tb", **kwd):
        self.sigma = sigma
        self.num_samples = num_samples
        self.instance = instance
        self.lr = lr
        self.logger_backend = logger_backend 
        self.seed = seed
        super().__init__(solver,lr,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])

        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.linalg.norm(y[i])

        loss = 0
        for i in range(len(normalized_y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma,noise='normal', batched= False)( normalized_y_hat[i] )
            loss += normalized_y[i].dot(sol[i] - op)
        loss /= len(normalized_y_hat)


        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret",train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss
        
    def validation_step(self, batch, batch_idx):
        x,y,sol,m = batch
    
        y_hat =  self(x).squeeze()
        val_regret = regret_fn(self.solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(self.solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)
       
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])

        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.linalg.norm(y[i])
       
        val_loss = 0
        for i in range(len(normalized_y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma, noise='normal', batched= False)( normalized_y_hat[i] )
            val_loss += normalized_y[i].dot(sol[i] - op)
        val_loss /= len(normalized_y_hat)
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}
        
        
class DPO_rp(baseline_mse):
    def __init__(self,solver,instance=1,sigma=0.1,num_samples=10, kappa=1, lr=1e-1,seed=0,scheduler=False, logger_backend="tb", **kwd):
        self.sigma = sigma
        self.num_samples = num_samples
        self.kappa = kappa
        self.instance = instance
        self.lr = lr
        self.logger_backend = logger_backend 
        self.seed = seed
        super().__init__(solver,lr,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        log_cost_norm(self, y_hat)
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./(1+torch.linalg.norm(y_hat[i])*1./self.kappa)

        loss = 0
        for i in range(len(normalized_y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma,noise='normal', batched= False)( normalized_y_hat[i] )
            loss += y[i].dot(sol[i] - op)
        loss /= len(normalized_y_hat)


        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_regret",train_regret, prog_bar=True, on_step=False, on_epoch=True, )
        return loss
        
    def validation_step(self, batch, batch_idx):
        x,y,sol,m = batch
    
        y_hat =  self(x).squeeze()
        val_regret = regret_fn(self.solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(self.solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)
       
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./(1+torch.linalg.norm(y_hat[i])*1./self.kappa) 
       
        val_loss = 0
        for i in range(len(normalized_y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma, noise='normal', batched= False)( normalized_y_hat[i] )
            val_loss += y[i].dot(sol[i] - op)
        val_loss /= len(normalized_y_hat)
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}