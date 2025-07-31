
from Trainer.NNModels import cora_net, cora_normednet, cora_nosigmoidnet
from Trainer.utils import regret_fn, regret_list, growpool_fn, abs_regret_fn
from Trainer.diff_layer import *
from DPO import perturbations
from DPO import fenchel_young as fy
from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution
import pandas as pd

import numpy as np 
from torch import nn, optim
from tqdm.auto import tqdm
import torch 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl


class baseline_mse(pl.LightningModule):
    def __init__(self,solver,lr=1e-1,mode='sigmoid',n_layers=2,seed=0,scheduler=False, **kwd):
        super().__init__()
        pl.seed_everything(seed)
        if mode=='sigmoid':
            self.model = cora_net(n_layers= n_layers)
            
        elif mode=="batchnorm" :
            self.model = cora_normednet(n_layers= n_layers)
        elif mode=="linear":
            self.model = cora_nosigmoidnet(n_layers= n_layers)
        self.mode= mode

        self.lr = lr
        self.solver = solver
        self.scheduler = scheduler
        self.save_hyperparameters("lr")

    def forward(self,x):
        return self.model(x) 
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
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

        if self.mode!= "sigmoid":
           y_hat = torch.sigmoid(y_hat)
        criterion2 = nn.BCELoss(reduction='mean')
        bceloss = criterion2(y_hat, sol)

        # val_norm = (criterion1(y_hat, torch.zeros(y_hat.shape))-1)*(criterion1(y_hat, torch.zeros(y_hat.shape))-1)
        # val_norm = (criterion1(y_hat, torch.zeros(y_hat.shape))-criterion1(y, torch.zeros(y.shape)))*(criterion1(y_hat, torch.zeros(y_hat.shape))-criterion1(y, torch.zeros(y.shape)))
        val_norm = criterion1(torch.var(y_hat,dim=1),torch.var(y,dim=1))
        
        self.log("val_norm", val_norm, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_bce", bceloss, prog_bar=True, on_step=False, on_epoch=True, )
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
        criterion2 = nn.BCELoss(reduction='mean')
        if self.mode!= "sigmoid":
           y_hat = torch.sigmoid(y_hat)

        bceloss = criterion2(y_hat, sol)
        self.log("test_regret", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("test_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )    
        self.log("test_bce", bceloss, prog_bar=True, on_step=False, on_epoch=True, )  
        self.log("test_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, )        
        return  {"test_regret": val_loss, "test_mse": mseloss}
    def predict_step(self, batch, batch_idx):
        '''
        I am using the the predict module to compute regret !
        '''
        solver = self.solver
        
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        regret_tensor = regret_list(solver,y_hat,y,sol,m)
        return regret_tensor
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

class baseline_bce(baseline_mse):
    def __init__(self,solver,lr=1e-1,mode='sigmoid',n_layers=2,seed=0,scheduler=False, **kwd):
            super().__init__(solver,lr,mode,n_layers,seed, scheduler) 
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        criterion = nn.BCELoss(reduction='mean')
        loss = criterion(y_hat,sol)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss   
       
        
class SPO(baseline_mse):
    def __init__(self,solver, lr=1e-1, alpha=2, mode='sigmoid',n_layers=2,seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.layer = SPOlayer(solver, alpha=alpha)
        # self.automatic_optimization = False
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss =  self.layer(y_hat, y,sol,m ) 
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class SPO_sd_norm(baseline_mse):
    def __init__(self,solver, lr=1e-1, alpha=2, mode='sigmoid',n_layers=2,seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.layer = SPOlayer(solver, alpha=alpha)
        # self.automatic_optimization = False
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.sqrt(torch.var(y_hat[i]))
        
        normalized_y = torch.zeros(y.shape)
        for i in range(len(y)):
            normalized_y[i] = y[i]*1./torch.sqrt(torch.var(y[i])) 
        
        loss =  self.layer(normalized_y_hat, normalized_y,sol,m ) 
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
class SPO_L2_norm(baseline_mse):
    def __init__(self,solver, lr=1e-1, alpha=2, mode='sigmoid',n_layers=2,seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.layer = SPOlayer(solver, alpha=alpha)
        # self.automatic_optimization = False
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.norm(y_hat[i])
        
        normalized_y = torch.zeros(y.shape)
        for i in range(len(y)):
            normalized_y[i] = y[i]*1./torch.norm(y[i])
        
        loss =  self.layer(normalized_y_hat, normalized_y,sol,m ) 
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DBB(baseline_mse):
    def __init__(self, solver,lr=1e-1,lambda_val=0.1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.layer = DBBlayer(solver,lambda_val=lambda_val)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        sol_hat  =  self.layer(y_hat, y,sol,m ) 
        loss = ((sol - sol_hat)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
        
class DBB_sd_norm(baseline_mse):
    def __init__(self, solver,lr=1e-1,lambda_val=0.1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.layer = DBBlayer(solver,lambda_val=lambda_val)
    

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        
        y_hat =  self(x).squeeze()

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./(torch.sqrt(torch.var(y_hat[i])))   

        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./(torch.sqrt(torch.var(y[i])))
        
        sol_hat = self.layer(normalized_y_hat, normalized_y,sol,m ) 
        
        loss = ((sol - sol_hat)*normalized_y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
class DBB_L2_norm(baseline_mse):
    def __init__(self, solver,lr=1e-1,lambda_val=0.1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.layer = DBBlayer(solver,lambda_val=lambda_val)
    

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        
        y_hat =  self(x).squeeze()

        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./(torch.linalg.norm(y_hat[i]))       
        
        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./(torch.linalg.norm(y[i]))
        
        sol_hat = self.layer(normalized_y_hat, normalized_y,sol,m ) 
        
        loss = ((sol - sol_hat)*normalized_y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        



class DPO(baseline_mse):
    def __init__(self,solver,sigma=0.1,num_samples=10, 
        lr=1e-1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        self.sigma = sigma
        self.num_samples = num_samples
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
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

        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DPO_sd_norm(baseline_mse):
    def __init__(self,solver,sigma=0.1,alpha=1,num_samples=10, 
        lr=1e-1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        self.sigma = sigma
        self.alpha = alpha
        self.num_samples = num_samples
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss = 0
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            normalized_y_hat[i] = y_hat[i]*1./(1e-5+torch.sqrt(torch.var(y_hat[i])))
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma, noise='normal', batched= False)( normalized_y_hat[i] )
            loss += y[i].dot(sol[i] - op)
        loss /= len(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DPO_L2_norm(baseline_mse):
    def __init__(self,solver,sigma=0.1,alpha=1,num_samples=10, 
        lr=1e-1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        self.sigma = sigma
        self.alpha = alpha
        self.num_samples = num_samples
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss = 0
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            normalized_y_hat[i] = y_hat[i]*1./(1e-5+torch.norm(y_hat[i]))
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma, noise='normal', batched= False)( normalized_y_hat[i] )
            loss += y[i].dot(sol[i] - op)
        loss /= len(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        



