# %%
from torch import nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer



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
y.shape,x.shape

# %%
learner = TitanicModel()

# %%
from torch.utils.data import Dataset, DataLoader

# %%
x_train=DataLoader(x)
y_train = DataLoader(y)

# %%
from fastai.data.core import DataLoaders

dls = DataLoaders(x_train, y_train)


# %%
learner=Learner(dls,model=learner,loss_func=CrossEntropyLossFlat(),metrics=accuracy)

# %%
learner.fit_one_cycle(5, lr_max=1e-3)

# %%



