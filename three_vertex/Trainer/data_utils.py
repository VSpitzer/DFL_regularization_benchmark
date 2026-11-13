import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from Trainer.instability_problem import InstabilityProblem

class Datawrapper():
    def __init__(self, x,y, M,  sols=None,solver=None):
        self.x = x
        self.y = y
        self.m = M
        if sols is not None:
            self.sols = sols
        else:
            y_iter = range(len(self.y))
            it = y_iter
            self.sols = np.array([solver.solve(self.y[i], self.m[i]) for i in it])
            self.sols = torch.from_numpy(self.sols).float()

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()
        self.m = torch.from_numpy(self.m).float()
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sols[index], self.m[index]


class DataModule(pl.LightningDataModule):
    def __init__(self,batch_size=10, generator=None,num_workers=8, seed=0, solver=None):
        super().__init__()

        with open('data/data_instability_problem.json') as f:
           data = json.load(f)

        x = data['inputs']
        y = data['outputs']
        m = data['params']

        x_train, y_train, m_train = np.array(x[:100]), np.array(y[:100]),  np.array(m[:100])
        x_valid, y_valid, m_valid = np.array(x[100:200]), np.array(y[100:200]), np.array(m[100:200])
        x_test, y_test, m_test = np.array(x[200:300]), np.array(y[200:300]), np.array(m[200:300])

        self.train_df = Datawrapper( x_train,y_train, m_train, solver=solver)
        self.valid_df  = Datawrapper( x_valid, y_valid, m_valid, solver=solver )
        self.test_df = Datawrapper( x_test, y_test, m_test, solver=solver )

        self.batch_size = batch_size
        self.generator = generator
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_df, batch_size=100,generator= self.generator, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_df, batch_size=100,generator= self.generator, num_workers=self.num_workers)
