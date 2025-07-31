import pandas as pd
import numpy as np 
from torch import nn, optim
from tqdm.auto import tqdm
import torch 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
# from Trainer.comb_solver import knapsack_solver, cvx_knapsack_solver,  intopt_knapsack_solver
from Trainer.comb_solver import knapsack_solver, cvx_knapsack_solver
from Trainer.utils import batch_solve, regret_fn, abs_regret_fn, regret_list,  growpool_fn
from Trainer.diff_layer import SPOlayer, SPOnormLayer, DBBlayer

from DPO import perturbations
from DPO import fenchel_young as fy
from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution

class baseline_mse(pl.LightningModule):
    def __init__(self,weights,capacity,n_items,lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__()
        pl.seed_everything(seed)
        self.model = nn.Linear(8,1)
        self.lr = lr
        self.solver = knapsack_solver(weights,capacity, n_items)
        self.scheduler = scheduler

    def forward(self,x):
        return self.model(x) 
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(self.solver, y_hat,y,sol)
        abs_val_loss= abs_regret_fn(self.solver, y_hat,y,sol)
        mseloss = criterion(y_hat, y)

        self.log("val_regret", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("abs_val_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
       
        return  {"val_regret": val_loss, "val_mse": mseloss, }
    def predict_step(self, batch, batch_idx):
        '''
        I am using the the predict module to compute regret !
        '''
        solver = self.solver
        
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        regret_tensor = regret_list(solver,y_hat,y,sol)
        return regret_tensor
    def test_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(self.solver, y_hat,y,sol)
        abs_val_loss= abs_regret_fn(self.solver, y_hat,y,sol)
        mseloss = criterion(y_hat, y)

        self.log("test_regret", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("abs_test_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("test_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
       
        return  {"test_regret": val_loss, "test_mse": mseloss, }
    def configure_optimizers(self):
        ############# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html ###

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
    def __init__(self,weights,capacity,n_items,lr=1e-1,alpha=2,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        self.layer = SPOlayer(self.solver, alpha=alpha)
    

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        
        y_hat =  self(x).squeeze()
        loss =  self.layer(y_hat, y,sol ) 
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
class SPO_sd_norm(baseline_mse):
    def __init__(self,weights,capacity,n_items,lr=1e-1,alpha=2,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        self.layer = SPOnormLayer(self.solver, alpha=alpha)
    

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        
        y_hat =  self(x).squeeze()
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.sqrt(torch.var(y_hat[i]))
        
        normalized_y = torch.zeros(y.shape)
        for i in range(len(y)):
            normalized_y[i] = y[i]*1./torch.sqrt(torch.var(y[i]))        
        
        loss =  self.layer(normalized_y_hat, normalized_y,sol ) 
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
class SPO_L2_norm(baseline_mse):
    def __init__(self,weights,capacity,n_items,lr=1e-1,alpha=2,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        self.layer = SPOnormLayer(self.solver, alpha=alpha)
    

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        
        y_hat =  self(x).squeeze()
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])
        
        normalized_y = torch.zeros(y.shape)
        for i in range(len(y)):
            normalized_y[i] = y[i]*1./torch.linalg.norm(y[i])       
        
        loss =  self.layer(normalized_y_hat, normalized_y,sol ) 
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DBB(baseline_mse):
    def __init__(self,weights,capacity,n_items,lambda_val=1., lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        self.layer = DBBlayer(self.solver, lambda_val=lambda_val)
    

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        
        y_hat =  self(x).squeeze()
        
        sol_hat = self.layer(y_hat, y,sol ) 
        
        loss = ((sol - sol_hat)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DBB_sd_norm(baseline_mse):
    def __init__(self,weights,capacity,n_items,lambda_val=1., lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        self.layer = DBBlayer(self.solver, lambda_val=lambda_val)
    

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        
        y_hat =  self(x).squeeze()

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.sqrt(torch.var(y_hat[i]))     

        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.sqrt(torch.var(y[i]))
        
        sol_hat = self.layer(normalized_y_hat, normalized_y,sol ) 
        
        loss = ((sol - sol_hat)*normalized_y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
class DBB_L2_norm(baseline_mse):
    def __init__(self,weights,capacity,n_items,lambda_val=1., lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        self.layer = DBBlayer(self.solver, lambda_val=lambda_val)
    

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        
        y_hat =  self(x).squeeze()

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])       
        
        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.linalg.norm(y[i])  
        
        sol_hat = self.layer(normalized_y_hat, normalized_y,sol ) 
        
        loss = ((sol - sol_hat)*normalized_y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DPO(baseline_mse):
    def __init__(self,weights,capacity,n_items,sigma=0.1,num_samples=10, lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)  

        fy_solver =  lambda y_: batch_solve(self.solver,y_) 
        self.criterion = fy.FenchelYoungLoss(fy_solver, num_samples= num_samples, 
        sigma= sigma,maximize = True, batched= True)

        @perturbations.perturbed(num_samples= num_samples, sigma= sigma, noise='normal',batched = True)
        def dpo_layer(y):
            return  batch_solve(self.solver,y)
        self.layer = dpo_layer

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        
        var = 0
        for i in range(len(y_hat)):
            var += torch.sqrt(torch.var(y_hat[i]))
        var *= 1./len(y_hat)
        
        sol_hat = self.layer(y_hat ) 
        loss = ((sol - sol_hat)*y).sum(-1).mean()  ## to minimize regret
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("avg_var",var, prog_bar=True, on_step=True, on_epoch=True, )
        return loss



class DPO_sd_norm(baseline_mse):
    def __init__(self,weights,capacity,n_items,sigma=0.1,num_samples=10, lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)  
        self.sigma = sigma
        self.num_samples = num_samples
        
        @perturbations.perturbed(num_samples= num_samples, sigma= sigma, noise='normal',batched = True)
        def dpo_layer(y):
            return  batch_solve(self.solver,y)
        self.layer = dpo_layer
        
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()

        var = 0
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.sqrt(torch.var(y_hat[i]))
            var += torch.sqrt(torch.var(y_hat[i]))
        var *= 1./len(y_hat)
        sol_hat = self.layer(normalized_y_hat) 

        loss = ((sol - sol_hat)*y).sum(-1).mean()  ## to minimize regret
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("avg_var",var, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
class DPO_L2_norm(baseline_mse):
    def __init__(self,weights,capacity,n_items,sigma=0.1,num_samples=10, lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)  
        self.sigma = sigma
        self.num_samples = num_samples
        
        @perturbations.perturbed(num_samples= num_samples, sigma= sigma, noise='normal',batched = True)
        def dpo_layer(y):
            return  batch_solve(self.solver,y)
        self.layer = dpo_layer
        
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()

        var = 0
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])
            var += torch.sqrt(torch.var(y_hat[i]))
        var *= 1./len(y_hat)
        sol_hat = self.layer(normalized_y_hat) 

        loss = ((sol - sol_hat)*y).sum(-1).mean()  ## to minimize regret
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("avg_var",var, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

