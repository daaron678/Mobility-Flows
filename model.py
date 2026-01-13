# pip install torch pandas numpy scikit-learn networkx gensim

import networkx as nx
import pandas as pd
from gensim.models import Word2Vec

df = pd.read_csv("mobility.csv")

G = nx.DiGraph()
for _, r in df.iterrows():
    G.add_edge(str(r.origin), str(r.dest), weight=r.flow)

# random walks
def random_walk(G, start, L=10):
    walk = [start]
    for _ in range(L):
        nbrs = list(G.neighbors(walk[-1]))
        if len(nbrs)==0: break
        walk.append(nbrs[np.random.randint(len(nbrs))])
    return walk

walks=[]
for node in G.nodes():
    for _ in range(10):
        walks.append(random_walk(G,node))

model = Word2Vec(walks, vector_size=64, window=5, sg=1, workers=4)
tract_embed = {n:model.wv[n] for n in G.nodes()}

import numpy as np

def build_token(row):
    o = tract_embed[str(row.origin)]
    d = tract_embed[str(row.dest)]
    features = np.array([
        row.flow,
        row.svi_o,
        row.svi_d,
        row.dist_hurricane,
        row.wind,
        row.rain
    ])
    return np.concatenate([o,d,features])   # 64+64+6 = 134 dims

sequences=[]
labels=[]

for day in sorted(df.day.unique())[:-1]:
    today = df[df.day==day]
    tomorrow = df[df.day==day+1]

    X=[build_token(r) for _,r in today.iterrows()]
    Y=[r.flow for _,r in tomorrow.iterrows()]

    sequences.append(X)
    labels.append(Y)

import torch
import torch.nn as nn

class MobilityTransformer(nn.Module):
    def __init__(self, dim=134):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4),
            num_layers=4
        )
        self.fc = nn.Linear(dim,1)

    def forward(self,x):
        z = self.encoder(x)
        return self.fc(z).squeeze(-1)

model = MobilityTransformer()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(30):
    for X,Y in zip(sequences,labels):
        X = torch.tensor(X).float().unsqueeze(1)
        Y = torch.tensor(Y).float()

        pred = model(X).squeeze()
        loss = loss_fn(pred[:len(Y)],Y)

        opt.zero_grad()
        loss.backward()
        opt.step()
