# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:59:10 2021

@author: tekin.evrim.ozmermer
"""

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class QSNN(nn.Module):
    def __init__(self,qubits):
        super(QSNN, self).__init__()
        
        self.pi = torch.tensor(np.pi)
        
        self.amplitude = torch.nn.Parameter(torch.randn(1,qubits))
        self.omega = torch.nn.Parameter(torch.randn(1,qubits))
        self.phase = torch.nn.Parameter(torch.randn(1,qubits))
        torch.nn.init.kaiming_normal_(self.amplitude, mode='fan_out')
        torch.nn.init.kaiming_normal_(self.omega, mode='fan_out')
        torch.nn.init.kaiming_normal_(self.phase, mode='fan_out')
        
        self.fc0 = nn.Linear(1, 1024, bias=False)
        torch.nn.init.kaiming_normal_(self.fc0.weight, mode='fan_out')
        self.fc1 = nn.Linear(1024, qubits, bias=False)
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        
        self.softmax = torch.nn.Softmax(dim=None)
    
    def forward(self, t):
        l = self.fc0(t)
        l = self.fc1(l)
        out = self.amplitude*torch.sin(2*self.pi*t*self.omega)/2
        out = out+0.5
        return out
    
    def to_state(self, field):
        # max_vals = torch.argmax(field, 1).int()
        # state = torch.zeros(field.shape).float()
        # for cnt,elm in enumerate(max_vals):
        #     state[cnt, elm] = 1
        state = torch.round(field)
        return state

qubits = 4
sample_size = 128
measured_states = torch.round(torch.rand(sample_size, qubits))
measured_time = torch.rand(sample_size,1)
dataset_ = torch.cat((measured_time,measured_states), dim = 1)
dataset_np = dataset_.numpy()

class QDataset(Dataset):
    def __init__(self, dataset_, transform=None):
        self.data = dataset_
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = QDataset(dataset_)
dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1, shuffle=False,
                                             num_workers=0)

model = QSNN(qubits)
criterion = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(lr = 0.01, params = model.parameters())

for epoch in range(2000):
    for sample in dataset_loader:
        field = model(sample[:,0].unsqueeze(1))
        state = model.to_state(field)
        loss = criterion(field, sample[:,1:])
        opt.zero_grad()
        loss.backward()
        print("EPOCH:", epoch)
        print("LOSS:", loss.item())
        opt.step()
        # print("|--->\nFIELD:\n\t{}\nSTATE:\n\t{}\n--->|\n".format(field,state))
        # print("STATE:", state)
        # print("TO-BE:", sample[:,1:])
        # print("\n")



