# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as p
import numpy as n
import pylab
import scipy.stats as stats
import networkx as nwx
import glob
import builtins 
from matplotlib.pyplot import margins
import os.path
import json
import sklearn.decomposition as dec

data2D = n.genfromtxt('tsne-vectors.tsv', delimiter=",")

ax = p.subplot(121)
ax.scatter(data2D[:,0], data2D[:,1], alpha=0.05, linewidth=0)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)

ax.get_xaxis().set_tick_params(which='both', top='off')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('tSNE')

data = n.genfromtxt('full-vectors.tsv', comments=None)
print(data.shape)
data = data[:,1:]

pca = dec.PCA(n_components=2)
pca.fit(data)
data = pca.transform(data)

ax = p.subplot(122)
ax.scatter(data[:,0], data[:,1], alpha=0.05, linewidth=0)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)

ax.get_xaxis().set_tick_params(which='both', top='off')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('PCA')

p.savefig('nobel.png')