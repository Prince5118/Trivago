
import csv
import copy

import pickle
import pprint
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.dataset import random_split

from collections import Counter
from tqdm import tqdm

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
item_df = pd.read_csv('data/item_metadata.csv')
submission_df = pd.read_csv('data/submission_popular.csv')




properties = []
for i in range(len(item_df)):
    properties += item_df['properties'][i].split("|")
property_count = Counter(properties)
property_set = list(property_count.keys())

onehot_df = pd.DataFrame(np.zeros([len(item_df), len(property_set)]), index=item_df['item_id'], columns=property_set)
for i, row in tqdm(item_df.iterrows()):
    item_id = row['item_id']
    properties = row['properties'].split("|")
    onehot_df.loc[item_id][properties] = 1

batch_size = 1024
num_epochs = 50
learning_rate = 5e-3
criterion = nn.L1Loss()

dataset = onehot_df.values
loader = torch.utils.data.DataLoader(dataset=torch.tensor(dataset), batch_size=batch_size, shuffle=True)

class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()
        self.enc = nn.Linear(157, 32)
        self.enc_act = nn.Tanh()
        self.dec = nn.Linear(32, 157)
        self.dec_act = nn.Sigmoid()

    def forward(self, x):
        encoded = self.enc_act(self.enc(x))
        decoded = self.dec_act(self.dec(encoded))
        return encoded, decoded


def fit(model,train_loader,learning_rate,num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.2)
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for i, data in enumerate(train_loader):
            item_meta = data.type(torch.FloatTensor).cuda()
            recon_item_meta = model(item_meta)[1]
            loss = criterion(recon_item_meta, item_meta)
            
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

simple_ae = SimpleAE().cuda()
fit(simple_ae, loader, learning_rate, num_epochs)


loader = torch.utils.data.DataLoader(dataset=torch.tensor(dataset), batch_size=batch_size, shuffle=False)
simple_ae.eval()
encoding_lst = []
for i, data in enumerate(loader):
    item_meta = data.type(torch.FloatTensor).cuda()
    encoding, decoding = simple_ae(item_meta)
    
    encoding = encoding.cpu().detach().numpy().tolist()
    encoding_lst += encoding
encoding_lst = np.array(encoding_lst)


item_encoding_dict = {}
for i, item_id in enumerate(onehot_df.index):
    item_encoding_dict[item_id] = encoding_lst[i]


with open("data/item_encoding_dict.pickle", "wb") as f:
    pickle.dump(item_encoding_dict, f)