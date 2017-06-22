# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as p
import numpy as n
import pylab
import scipy.stats as stats
import networkx as nwx
import glob, sys
import builtins, re
from matplotlib.pyplot import margins
import os.path
import json
import sklearn.decomposition as dec
import random
from sklearn.manifold import TSNE

import pandas as pd

data = pd.read_csv('full-vectors.tsv', skiprows=[0], delim_whitespace=True, header=None)
names = pd.read_csv('names.csv', header=None)

nms = []
instances = n.zeros(shape=(names.shape[0], data.shape[1]))
print(data.shape)

for i, (index, name) in names.iterrows():
    instances[i, :] = data.values[data.values[:, 0] == index, :] 
    nms.append(name)

print(instances)
print(nms)

instances = instances[:, 1:]

print(instances.min(), instances.max())

# PCA
pca = dec.PCA(n_components=2)
pca.fit(instances)

instancesPCA = pca.transform(instances)

p.figure(figsize=(16,16))
ax = p.subplot(111)
ax.scatter(instancesPCA[:,0], instancesPCA[:,1], linewidth=0, c='r')

for i in range(len(nms)):
    if random.random() < 1.0:
        srch = re.search('cellextradry_process_(.*)_step_(.*)_mixture_(.*)', nms[i])
        label = 'p{}s{}m{}'.format(srch.group(1), srch.group(2), srch.group(3))
        ax.annotate(label, xy=(instancesPCA[i,:]), xytext=(5,0),
        textcoords='offset points', 
        ha='left' if bool(random.getrandbits(1)) else 'right', 
        va='bottom')

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)

ax.get_xaxis().set_tick_params(which='both', top='off')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('PCA')

p.savefig('micro-organisms.pca.png')

# tSNE

for perp in [2,5,30,50]:
    model = TSNE(n_components=2, random_state=0, perplexity=perp, method='exact')
    n.set_printoptions(suppress=True)

    instancesTSNE = model.fit_transform(instances)
    
    p.figure(figsize=(16,16))
    ax = p.subplot(111)
    ax.scatter(instancesTSNE[:,0], instancesTSNE[:,1], linewidth=0, c='r')

    for i in range(len(nms)):
        if random.random() < .1:
            srch = re.search('cellextradry_process_(.*)_step_(.*)_mixture_(.*)', nms[i])
            label = 'p{}s{}m{}'.format(srch.group(1), srch.group(2), srch.group(3))
            ax.annotate(label, xy=(instancesTSNE[i,:]), xytext=(5,0),
            textcoords='offset points', 
            ha='left' if bool(random.getrandbits(1)) else 'right', 
            va='bottom')
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    
    ax.get_xaxis().set_tick_params(which='both', top='off')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('tSNE, perplexity: {}'.format(perp))
    
    p.savefig('micro-organisms.tsne.{}.png'.format(perp))


print('done')