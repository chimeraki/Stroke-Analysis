########### Trajectory Profile Clustering and Graph Neural Networks for Stroke Recovery Analysis #########
###############Sanjukta Krishnagopal####################
#################sanju33@gmail.com######################
###################August 2021##########################


import numpy as np
from matplotlib.pyplot import *
import csv
from numpy import *
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import math
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
import pandas as pd
import community
import copy
import dgl
import torch
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.spatial import distance_matrix
matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13)

#form minibatches from graphs
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


#################load and preprocess data
df=pd.read_csv('ninds_updated_all.csv', sep=',',header=None).iloc[1:]

var = ['ATAXIA','CONSCIO','DYSAR','EXTIN', 'GAZE', 'LANG', 'LOCCOM','LOCQU','MOTORLA','MOTORLL', 'MOTORRA', 'MOTORRL', 'PALSY','SENSORY','VISUAL'] #15 total variables

ranges = [2,3,2,2,2,2,2,2,4,4,4,4,3,2,3]

dat = df[df.columns[[119,134,149,164,179]]].to_numpy(dtype = float)[:,:,np.newaxis] #sataxia

col_num= [[110,125,140,155,170], [123,138,153,168,183],[124,139,154,169,184],[113,128,143,158,173] , [122,137,152,167,182],[112,127,142,157,172],[111,126,141,156,171],[116,131,146,161,176],[118,133,148,163,178],[117,132,147,162,177],[119,134,149,164,179],[115,130,145,160,175],[121,136,151,166,181],[114,129,144,159,174]]
col_num=np.array(col_num)-1

for i in col_num:
   b = df[df.columns[i]].to_numpy(dtype = float)[:,:,np.newaxis] 
   dat  = np.append(dat, b, axis = 2)

#################removing dead and incomplete patients
ded =df[df.columns[[213]]].to_numpy(dtype = float)[:,:,np.newaxis]
imp = [192,193,194,195]
imputed_cols=df[df.columns[imp]].to_numpy(dtype = float)[:,:,np.newaxis][:,:,0]
imp_pt=np.where(np.max(imputed_cols, axis = 1))[0]
ded_pt=np.where(ded[:,0,0]==1)[0]
exclude = np.union1d(imp_pt,ded_pt)
print ('Total number of patients excluded :' +str(len(exclude)))

############### variable determining whether they got treatment
treat =df[df.columns[[12]]].to_numpy(dtype = float)[:,:,np.newaxis]

dat = np.delete(dat, exclude, axis=0) #remove dead and imputed patients
treat = np.delete(treat, exclude, axis=0)

treat1 = np.where(treat==1)[0]
treat2 = np.where(treat==2)[0]

dat_treat1 = dat[treat1, :,:]
dat_treat2 = dat[treat2, :,:]

#dat = dat_treat1 #uncomment if you want to find subsets based just on treated patients or untreated patients.

#dat = dat_treat2 #if you want to separate the treated from non treated patients
var_no=shape(dat)[2]
pat_no=shape(dat)[0]

#threshold trajectory profile matrix to get 'severe' disease only for community detection
z = copy.deepcopy(dat[:,:,:])
th_frac = 0.5
th= np.array(ranges)*th_frac #threshold the value
th_p = th
z[z<th]=0
z[z>=th]=1

###############generate a variable-interaction matrix per timepoint per patient
g = np.zeros((shape(dat)[0],shape(dat)[1],shape(dat)[2],shape(dat)[2]))
for p in range(shape(dat)[0]):
   for t in range(shape(dat)[1]):
      s = z[p,t][:, np.newaxis]
      #s = np.array(dat[p,t])[:,None]
      g[p,t]= 1 - distance_matrix(s,s)



###training and test set for the graph neural network
tr = int(np.shape(g)[0]*0.7)
te0 = tr
te = tr
g_train = g[:tr,:,:]
g_test = g[tr:,:,:]
treat_train = treat[:tr]
treat_test = treat[tr:]
pat_no_train=shape(g_train)[0]




################first order Louvain community detection on the patient-patient matrix
M = np.zeros((pat_no,pat_no))
for i in range(pat_no):
   for j in range(pat_no):
      M[i][j]=(z[i]==z[j]).sum() #use this for clusetring using bipartite network
      

H=nx.Graph(M)              
part = community.best_partition(H)
values = [part.get(node) for node in H.nodes()]
mod = community.modularity(part,H)

pat_indices = np.arange(pat_no)


# aggregate patient profile for each community
prof=[]
no_of_ppl =[]
treat_profiles = []
uni=np.unique(values)
for comm in uni:
   s = np.where(values == comm)[0]
   prof.append(np.mean(z[s],axis=0))
   treat_profiles.append(treat[s])
   no_of_ppl.append(len(s))
prof=np.array(prof)
treat_profiles = np.array(treat_profiles)
print (no_of_ppl, ':number of ppl in each comm')
print (len(no_of_ppl), ':number of communities')


#delete communities with less than a threshold number of patients
rem = np.where(np.array(no_of_ppl) < 10)[0]
prof=np.delete(prof, rem, axis=0)
no_of_ppl=np.delete(no_of_ppl, rem, axis=0)
comm_no = shape(prof)[0]
for i in range(len(rem)):
    s = np.where(values == rem[i])[0]
    tr-= np.count_nonzero(s<tr)
    te+= np.count_nonzero(s>te)
    pat_indices = np.delete(pat_indices, s)
    values = np.delete(values, s)

pat_comm = dict(zip(pat_indices, values))

#######################plotting the community profile
sort_index = np.argsort(no_of_ppl)
no_of_ppl=no_of_ppl[sort_index]
prof=prof[sort_index]
treat_profiles = treat_profiles[sort_index]

figure()
fig, axes = subplots(nrows=comm_no, ncols=1) #if you need to plot cbar, plot here and then crop out common colorbar
i=0
for ax in axes.flat:
    ax.set_yticks(range(0,5))
    ax.set_yticklabels(['bl','2hr','24hr','7-10d','3mth'])
    trt_no = round(np.sum(treat_profiles[i] ==1)/len(treat_profiles[i]),2)
    im = ax.imshow(prof[i],cmap='Greys',vmin=0,vmax=1)
    ax.set_ylabel(str(no_of_ppl[i])+ 'patnt \n ',fontsize=12)
    if i!=comm_no-1:
       ax.set_xticks([])
       i+=1
    else:
       #im = ax.imshow(np.mean(prof, axis = 0),cmap='Greys',vmin=0,vmax=1)
       #ax.set_ylabel('Time \n All patnt',fontsize=12)
       ax.set_xticks(np.arange(len(var)))
       ax.set_xticklabels(var, rotation=90,fontsize=14)

xlabel('Variables',fontsize=16)
fig.subplots_adjust(right=1.0)
cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
savefig('subtypes_TP1_thresholding_val_'+str(th_frac)+'.pdf', bbox_inches='tight')



################# Graph classification using variable interaction network
# Create model
model = Classifier(1, 64, len(no_of_ppl))
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
accuracy = []

#iterate over timesteps
for t in range(5):
    model.train()
    epoch_losses = []
    for epoch in range(25):
        epoch_loss = 0
        for iter, (pat_ind, label) in enumerate(zip(pat_indices[:tr], values[:tr])):
            g= dgl.from_networkx(nx.from_numpy_matrix(g_train[pat_ind,t]))
            #g= dgl.from_networkx(nx.from_numpy_matrix(np.mean(g_train[pat_ind],axis=0)))
            g = dgl.add_self_loop(g)
            prediction = model(g)
            loss = loss_func(prediction, torch.from_numpy(np.array([label])))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    # The learning curve of a run is presented below.

    '''plt.title('cross entropy averaged over minibatches')
    plt.plot(epoch_losses)
    plt.show()'''

    ############# test on the test set
    model.eval()
    success=0
    fail=0
    # Convert a list of tuples to two lists
    for iter, (pat_ind, label) in enumerate(zip(pat_indices[te:], values[te:])):
       g=dgl.from_networkx(nx.from_numpy_matrix(g_test[pat_ind-te0,t]))
       #g=dgl.from_networkx(nx.from_numpy_matrix(np.mean(g_test[pat_ind-te0],axis=0)))
       g = dgl.add_self_loop(g)
       probs_Y = model(g)
       pred_t=torch.softmax(probs_Y, dim=1)
       if int(torch.argmax(pred_t)) == label:
           success+=1
           print ('success')
       else:
           fail+=1
           print ('fail')

    #probs_Y = torch.softmax(model(test_bg), 1)
    print('At time: ' + str(t) + ' accuracy on test set: {:.4f}%'.format(
        success/(success+fail)))
    accuracy.append(success/(success+fail))

#############plotting histograms of number of people severely affected by variables

'''#calculate the number of 'severe' diseases in individuals:
severe_count = [None for _ in range(5)]
for i in range(5):
   severe_count[i] = np.count_nonzero(z[:,i,:], axis = 1)

figure()
hist(severe_count, label = ['baseline','2hr','24hr','7-10dy','3mth'])
legend()
xticks(np.arange(len(var)),var, rotation=90,fontsize=14)
savefig('distribution_TP1_val'+str(th_frac)+'.pdf', bbox_inches='tight')'''


####plot accuracy for TP with/without and all
timestamps  = ['Baseline','2 hr','24 hr','7-10 dy','3 mth']
figure()
plot(tp1acc, label = 'Treated')
plot(tp2acc, label = 'Untreated')
plot(tpall, label = 'All')
legend(fontsize = 14)
xticks(np.arange(len(timestamps)),timestamps, rotation=90,fontsize=14)
xlabel('Timestamp', fontsize = 16)
ylabel('Accuracy', fontsize = 16)
savefig('Accuracy_plots_binarized.pdf', bbox_inches = 'tight')
