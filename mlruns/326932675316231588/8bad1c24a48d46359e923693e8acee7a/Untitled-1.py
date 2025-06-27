# %%
from torch import nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
import torch


# %%
class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.li = nn.Linear(10,10)
        self.li2 = nn.Linear(10,1)
        self.libi= nn.Sigmoid()

    def forward(self,x):  
        out = self.li(x)
        out=self.li2(out)
        out = self.libi(out)
        return out  

# %%
from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.losses import CrossEntropyLossFlat  # for classification




# %%
df = pd.read_csv("Titanic-Dataset.csv")

# %%
x=torch.randn(100,10)
y=torch.randn(100,1)

# %%
from torch.utils.data import Dataset, DataLoader

# %%
class TitanicDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        self.x= x.float()
        self.y = y.float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]  # ✅ tuple, not dict
 

# %%
dataloader=DataLoader(TitanicDataset(x,y))
from fastai.data.core import DataLoaders

dls = DataLoaders(dataloader)

# %%

model = TitanicModel()

# %%
learn=Learner(dls,model=model,loss_func=CrossEntropyLossFlat(),metrics=accuracy)

# %%
learn.fit_one_cycle(5, lr_max=1e-3)

# %%



