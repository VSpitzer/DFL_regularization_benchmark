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

    


class DPO_reg(baseline_mse):
    def __init__(self,solver,instance=1,sigma=0.1,kappa=1, num_samples=10, lr=1e-1,seed=0,scheduler=False, logger_backend="tb", **kwd):
        self.kappa = kappa
        self.sigma = sigma
        self.instance=instance
        self.num_samples = num_samples
        self.lr = lr
        self._train_out_norms = []
        self._train_sol = []
        self._train_grads = []
        self._validation_out_norms = []
        self._validation_sol = []
        self.logger_backend = logger_backend 
        self.seed = seed
        super().__init__(solver,lr,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)
        
        loss = 0
        per_sample_loss = torch.zeros(len(y_hat))
        per_sample_sol = torch.zeros(len(y_hat))
        norm=0
        max_norm=0
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            normalized_y_hat[i] = y_hat[i]*1./(1+torch.linalg.norm(y_hat[i])*1./self.kappa)    
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma, noise='normal', batched= False)( normalized_y_hat[i] )
            loss += y[i].dot(sol[i] - op)
            per_sample_loss[i] = y[i].dot(sol[i] - op)
            per_sample_sol[i] = self.solver.get_vertex_solve(y_hat[i].detach().numpy(),m[i].detach().numpy())
            norm += torch.norm(normalized_y_hat[i])
            max_norm = max(max_norm,torch.norm(normalized_y_hat[i]))
        loss /= len(y_hat)
        norm /= len(y_hat)
        
        out_norms = y_hat.flatten(1).norm(p=2, dim=1)
        grad_yhat = torch.autograd.grad(
            per_sample_loss.sum(),
            y_hat,
            retain_graph=True,   # so Lightning can still call backward on 'loss'
            create_graph=False
        )[0]  
        grads = grad_yhat.flatten(1)
        self._train_out_norms.append(out_norms.detach())
        self._train_sol.append(per_sample_sol.detach())
        self._train_grads.append(grads.detach().numpy())
        
        self.log("train_norm",norm, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_max_norm",max_norm, prog_bar=True, on_step=False, on_epoch=True, )
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

        per_sample_sol = torch.zeros(len(y_hat))
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            per_sample_sol[i] = self.solver.get_vertex_solve(y_hat[i].detach().numpy(),m[i].detach().numpy())
        
        out_norms = y_hat.flatten(1).norm(p=2, dim=1)
        self._validation_out_norms.append(out_norms.detach())
        self._validation_sol.append(per_sample_sol.detach())
        
        val_loss = 0
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            normalized_y_hat[i] = y_hat[i]*1./(1+torch.linalg.norm(y_hat[i])*1./self.kappa)    
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma, noise='normal', batched= False)( normalized_y_hat[i] )
            val_loss += y[i].dot(sol[i] - op)
        val_loss /= len(y_hat)
       
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", val_regret, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
        return  {"val_loss": val_loss, "val_regret": val_regret}
        
    def on_train_epoch_end(self):
        import numpy as np

        if not self.logger:
            # cleanup even if no logger
            self._train_out_norms.clear()
            self._train_sol.clear()
            self._train_grads.clear()
            return

        out_norms = torch.cat(self._train_out_norms, dim=0)  # (N_epoch,)
        sol = torch.cat(self._train_sol, dim=0)  # (N_epoch,)
        grads = self._train_grads  

        # Log only on global rank 0 to avoid duplicates
        global_rank = getattr(self.trainer, "global_rank", 0)
        if global_rank == 0:
            epoch = self.current_epoch
            # ---- TensorBoard ----
            if self.logger_backend == "tb":
                tb = self.logger.experiment  # SummaryWriter
                tb.add_histogram("train/output_norm", out_norms, global_step=epoch)
                tb.add_histogram("train/sol", sol, global_step=epoch)

                # also save the raw arrays as .npz inside the log dir (artifact-like)
                log_dir = getattr(self.logger, "Rslt/samples_data/", "Rslt/samples_data/")
                np.savez(
                    f"{log_dir}/DPO_reg_train_per_sample_norms_inst{self.instance}_seed{self.seed}_lr{self.lr}_sigma{self.sigma}_kappa{self.kappa}_epoch{epoch}.npz",
                    output_norm=out_norms.numpy(),
                    sol=sol.numpy(),
                    grads=grads,
                )

        # cleanup to free memory before next epoch
        self._train_out_norms.clear()
        self._train_sol.clear()
        self._train_grads.clear()

        return
        
    def on_validation_epoch_end(self):
        import numpy as np

        if not self.logger:
            # cleanup even if no logger
            self._validation_out_norms.clear()
            self._validation_sol.clear()
            return

        out_norms = torch.cat(self._validation_out_norms, dim=0)  # (N_epoch,)
        sol = torch.cat(self._validation_sol, dim=0)  # (N_epoch,)

        # Log only on global rank 0 to avoid duplicates
        global_rank = getattr(self.trainer, "global_rank", 0)
        if global_rank == 0:
            epoch = self.current_epoch
            # ---- TensorBoard ----
            if self.logger_backend == "tb":
                tb = self.logger.experiment  # SummaryWriter
                tb.add_histogram("train/output_norm", out_norms, global_step=epoch)
                tb.add_histogram("train/sol", sol, global_step=epoch)

                # also save the raw arrays as .npz inside the log dir (artifact-like)
                log_dir = getattr(self.logger, "Rslt/samples_data/", "Rslt/samples_data/")
                np.savez(
                    f"{log_dir}/DPO_reg_validation_per_sample_norms_inst{self.instance}_seed{self.seed}_lr{self.lr}_sigma{self.sigma}_kappa{self.kappa}_epoch{epoch}.npz",
                    output_norm=out_norms.numpy(),
                    sol=sol.numpy(),
                )

        # cleanup to free memory before next epoch
        self._validation_out_norms.clear()
        self._validation_sol.clear()

        return

class DPO(baseline_mse):
    def __init__(self,solver,instance=1,sigma=0.1,num_samples=10, lr=1e-1,seed=0,scheduler=False, logger_backend="tb", **kwd):
        self.sigma = sigma
        self.num_samples = num_samples
        self.instance = instance
        self.lr = lr
        self._train_out_norms = []
        self._train_sol = []
        self._train_grads = []
        self._validation_out_norms = []
        self._validation_sol = []
        self.logger_backend = logger_backend 
        self.seed = seed
        super().__init__(solver,lr,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        train_regret= regret_fn(self.solver,y_hat,y,sol,m)
        
        loss = 0
        per_sample_loss = torch.zeros(len(y_hat))
        per_sample_sol = torch.zeros(len(y_hat))
        norm=0
        max_norm=0
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma,noise='normal', batched= False)( y_hat[i] )
            per_sample_loss[i] = y[i].dot(sol[i] - op)
            per_sample_sol[i] = self.solver.get_vertex_solve(y_hat[i].detach().numpy(),m[i].detach().numpy())
            loss += y[i].dot(sol[i] - op)
            norm += torch.norm(y_hat[i])
            max_norm = max(max_norm,torch.norm(y_hat[i]))
        loss /= len(y_hat)
        norm /= len(y_hat)
        
        out_norms = y_hat.flatten(1).norm(p=2, dim=1)
        grad_yhat = torch.autograd.grad(
            per_sample_loss.sum(),
            y_hat,
            retain_graph=True,   # so Lightning can still call backward on 'loss'
            create_graph=False
        )[0]  
        grads = grad_yhat.flatten(1)
        self._train_out_norms.append(out_norms.detach())
        self._train_sol.append(per_sample_sol.detach())
        self._train_grads.append(grads.detach().numpy())

        
        self.log("train_norm",norm, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("train_max_norm",max_norm, prog_bar=True, on_step=False, on_epoch=True, )
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

        per_sample_sol = torch.zeros(len(y_hat))
        normalized_y_hat = torch.zeros(y_hat.shape)
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            per_sample_sol[i] = self.solver.get_vertex_solve(y_hat[i].detach().numpy(),m[i].detach().numpy())
        
        out_norms = y_hat.flatten(1).norm(p=2, dim=1)
        self._validation_out_norms.append(out_norms.detach())
        self._validation_sol.append(per_sample_sol.detach())
       
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
        
    def on_train_epoch_end(self):
        import numpy as np

        if not self.logger:
            # cleanup even if no logger
            self._train_out_norms.clear()
            self._train_sol.clear()
            self._train_grads.clear()
            return

        out_norms = torch.cat(self._train_out_norms, dim=0)  # (N_epoch,)
        sol = torch.cat(self._train_sol, dim=0)  # (N_epoch,)
        grads = self._train_grads  

        # Log only on global rank 0 to avoid duplicates
        global_rank = getattr(self.trainer, "global_rank", 0)
        if global_rank == 0:
            epoch = self.current_epoch
            # ---- TensorBoard ----
            if self.logger_backend == "tb":
                tb = self.logger.experiment  # SummaryWriter
                tb.add_histogram("train/output_norm", out_norms, global_step=epoch)
                tb.add_histogram("train/sol", sol, global_step=epoch)

                # also save the raw arrays as .npz inside the log dir (artifact-like)
                log_dir = getattr(self.logger, "Rslt/samples_data/", "Rslt/samples_data/")
                np.savez(
                    f"{log_dir}/DPO_train_per_sample_norms_inst{self.instance}_seed{self.seed}_lr{self.lr}_sigma{self.sigma}_epoch{epoch}.npz",
                    output_norm=out_norms.numpy(),
                    sol=sol.numpy(),
                    grads=grads,
                )

        # cleanup to free memory before next epoch
        self._train_out_norms.clear()
        self._train_sol.clear()
        self._train_grads.clear()

        return

    def on_validation_epoch_end(self):
        import numpy as np

        if not self.logger:
            # cleanup even if no logger
            self._validation_out_norms.clear()
            self._validation_sol.clear()
            return

        out_norms = torch.cat(self._validation_out_norms, dim=0)  # (N_epoch,)
        sol = torch.cat(self._validation_sol, dim=0)  # (N_epoch,)

        # Log only on global rank 0 to avoid duplicates
        global_rank = getattr(self.trainer, "global_rank", 0)
        if global_rank == 0:
            epoch = self.current_epoch
            # ---- TensorBoard ----
            if self.logger_backend == "tb":
                tb = self.logger.experiment  # SummaryWriter
                tb.add_histogram("train/output_norm", out_norms, global_step=epoch)
                tb.add_histogram("train/sol", sol, global_step=epoch)

                # also save the raw arrays as .npz inside the log dir (artifact-like)
                log_dir = getattr(self.logger, "Rslt/samples_data/", "Rslt/samples_data/")
                np.savez(
                    f"{log_dir}/DPO_validation_per_sample_norms_inst{self.instance}_seed{self.seed}_lr{self.lr}_sigma{self.sigma}_epoch{epoch}.npz",
                    output_norm=out_norms.numpy(),
                    sol=sol.numpy(),
                )

        # cleanup to free memory before next epoch
        self._validation_out_norms.clear()
        self._validation_sol.clear()

        return