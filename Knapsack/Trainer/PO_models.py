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


class ADPO_norm(baseline_mse):
    def __init__(self,weights,capacity,n_items, sigma_max=1., sigma_min=0.001, num_samples=10, alpha=1., N_periods=1, len_periods=1, lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)  
        self.num_samples = num_samples
        
        self.sigma_tab = np.zeros(N_periods*len_periods)
        for i in range(N_periods):
            self.sigma_tab[i*len_periods:(i+1)*len_periods] = np.array([sigma_min + (sigma_max-sigma_min)*(1-j*1./len_periods) for j in range(len_periods)])
        
        # self.sigma_tab = np.concatenate((np.linspace(1,5,20),np.ones(10)*5))
        # self.sigma_tab = np.ones(30)

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
           
        @perturbations.perturbed(num_samples= self.num_samples, sigma= self.sigma_tab[self.current_epoch], noise='normal',batched = True)
        def dpo_layer(y):
            return  batch_solve(self.solver,y)
        self.layer = dpo_layer
           
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            # normalized_y_hat[i] = y_hat[i]*1./torch.sqrt(torch.var(y_hat[i]))
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])
        sol_hat = self.layer(normalized_y_hat) 

        loss = ((sol - sol_hat)*y).sum(-1).mean()  ## to minimize regret,
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

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
        
class SPO_norm(baseline_mse):
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
        
class SPO_norm_2(baseline_mse):
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

class DBB_norm(baseline_mse):
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
        
class DBB_norm_2(baseline_mse):
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

class FenchelYoung(baseline_mse):
    def __init__(self,weights,capacity,n_items,sigma=0.1,num_samples=10, lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)  

        fy_solver =  lambda y_: batch_solve(self.solver,y_) 
        self.criterion = fy.FenchelYoungLoss(fy_solver, num_samples= num_samples, 
        sigma= sigma, noise='normal', maximize = True, batched= True)
        self.sigma = sigma
        
    def training_step(self, batch, batch_idx):
        criterion = self.criterion 
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.sqrt(torch.var(y_hat[i])) 
        
        var = torch.mean(torch.sqrt(torch.var(y_hat, dim=1)))
        
        loss = criterion(normalized_y_hat,sol).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("avg_var",var, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DPO_norm(baseline_mse):
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
        
class DPO_norm_2(baseline_mse):
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

class IMLE(baseline_mse):
    def __init__(self,weights,capacity,n_items, k=5, nb_iterations=100,nb_samples=5, beta=10.0,
            temperature=1.0,   lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        imle_solver = lambda y_: batch_solve(self.solver,y_)

        target_distribution = TargetDistribution(alpha=1.0, beta=beta)
        noise_distribution = SumOfGammaNoiseDistribution(k= k, nb_iterations= nb_iterations)

        self.layer = imle(imle_solver,  target_distribution=target_distribution,noise_distribution=noise_distribution,
                    input_noise_temperature= temperature, target_noise_temperature= temperature,
                    nb_samples= nb_samples)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        sol_hat = self.layer(y_hat ) 
        loss = ((sol - sol_hat)*y).sum(-1).mean()  ## to minimze regret
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
class IMLE_norm(baseline_mse):
    def __init__(self,weights,capacity,n_items, k=5, nb_iterations=100,nb_samples=5, beta=10.0,
            temperature=1.0,   lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        imle_solver = lambda y_: batch_solve(self.solver,y_)

        target_distribution = TargetDistribution(alpha=1.0, beta=beta)
        noise_distribution = SumOfGammaNoiseDistribution(k= k, nb_iterations= nb_iterations)

        self.layer = imle(imle_solver,  target_distribution=target_distribution,noise_distribution=noise_distribution,
                    input_noise_temperature= temperature, target_noise_temperature= temperature,
                    nb_samples= nb_samples)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.sqrt(torch.var(y_hat[i]))
        sol_hat = self.layer(normalized_y_hat) 
        
        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.sqrt(torch.var(y[i]))
        
        loss = ((sol - sol_hat)*normalized_y).sum(-1).mean()  ## to minimze regret
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
class IMLE_norm_2(baseline_mse):
    def __init__(self,weights,capacity,n_items, k=5, nb_iterations=100,nb_samples=5, beta=10.0,
            temperature=1.0,   lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        imle_solver = lambda y_: batch_solve(self.solver,y_)

        target_distribution = TargetDistribution(alpha=1.0, beta=beta)
        noise_distribution = SumOfGammaNoiseDistribution(k= k, nb_iterations= nb_iterations)

        self.layer = imle(imle_solver,  target_distribution=target_distribution,noise_distribution=noise_distribution,
                    input_noise_temperature= temperature, target_noise_temperature= temperature,
                    nb_samples= nb_samples)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            normalized_y_hat[i] = y_hat[i]*1./torch.linalg.norm(y_hat[i])
        sol_hat = self.layer(normalized_y_hat) 
        
        normalized_y = torch.zeros(y.shape)
        for i in range(len(y_hat)):
            normalized_y[i] = y[i]*1./torch.linalg.norm(y[i])
        
        loss = ((sol - sol_hat)*normalized_y).sum(-1).mean()  ## to minimze regret
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
        
class DCOL(baseline_mse):
    '''
    Implementation oF QPTL using cvxpyayers
    '''
    def __init__(self,weights,capacity,n_items,mu=1.,lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        self.comblayer = cvx_knapsack_solver(weights,capacity,n_items,mu=mu)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        sol_hat = self.comblayer(y_hat)
        loss = ((sol - sol_hat)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss    




class IntOpt(baseline_mse):
    def __init__(self,weights,capacity,n_items,thr=0.1,damping=1e-3,lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        self.comblayer = intopt_knapsack_solver(weights,capacity,n_items, thr= thr,damping= damping)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        sol_hat = self.comblayer(y_hat)
        loss = ((sol - sol_hat)*y).sum(-1).mean()

        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss  

from Trainer.CacheLosses import *
class CachingPO(baseline_mse):
    def __init__(self, weights,capacity,n_items,init_cache,tau=1.,growth=0.1,loss="listwise",lr=1e-1,seed=0,scheduler=False, **kwd):
        super().__init__(weights,capacity,n_items,lr,seed, scheduler)
        '''tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
        '''


        if loss=="pointwise":
            self.loss_fn = PointwiseLoss()
        elif loss=="pairwise":
            self.loss_fn = PairwiseLoss(margin=tau)
        elif loss == "pairwise_diff":
            self.loss_fn = PairwisediffLoss()
        elif loss == "listwise":
            self.loss_fn = ListwiseLoss(temperature=tau)
        elif loss== 'NCE':
            self.loss_fn = NCE()
        elif loss== 'MAP':
            self.loss_fn = MAP()
        elif loss== 'NCE_c':
            self.loss_fn = NCE_c()
        elif loss== 'MAP_c':
            self.loss_fn = MAP_c()
        elif loss== 'MAP_c_actual':
            self.loss_fn = MAP_c_actual()
        else:
            raise Exception("Invalid Loss Provided")

        self.growth = growth
        cache_np = init_cache.detach().numpy()
        cache_np = np.unique(cache_np,axis=0)
        # torch has no unique function, so we have to do this
        init_cache =  torch.from_numpy(cache_np).float()
        self.cache = init_cache
    

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        
        if (np.random.random(1)[0]< self.growth) or len(self.cache)==0:
            self.cache= growpool_fn(self.solver,self.cache, y_hat)

        loss = self.loss_fn(y_hat,y,sol,self.cache)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss 

