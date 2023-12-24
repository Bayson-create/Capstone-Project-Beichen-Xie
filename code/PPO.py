from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
            try:
                    np_array = np_array.astype(dtype)
            except:
                return np_array.float()
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


# def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
#     if done:
#         v_s_ = 0.               # terminal
#     else:
#         v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

#     buffer_v_target = []
#     for r in br[::-1]:    # reverse buffer r
#         v_s_ = r + gamma * v_s_
#         buffer_v_target.append(v_s_)
#     buffer_v_target.reverse()

#     loss = lnet.loss_func(
#         v_wrap(np.vstack(bs)),
#         v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
#         v_wrap(np.array(buffer_v_target)[:, None]))

#     # calculate local gradients and push local parameters to global
#     opt.zero_grad()
#     loss.backward()
#     for lp, gp in zip(lnet.parameters(), gnet.parameters()):
#         gp._grad = lp.grad
#     opt.step()

#     # pull global parameters
#     lnet.load_state_dict(gnet.state_dict())

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma, old_probs):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    #print(np.vstack(bs))
    #print(np.vstack(ba))
    #print(buffer_v_target)
    #print(np.vstack(bs))
    to_loss_s=v_wrap(np.vstack(bs))
    to_loss_a= v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba))
    #to_loss_b=v_wrap(np.array(buffer_v_target)[:, None])

    to_loss_b=v_wrap(np.vstack(br))

    #print(to_loss_s.shape)
    #print(to_loss_a.shape)
    #print(to_loss_b.shape)


    loss = lnet.loss_func(to_loss_s, to_loss_a, to_loss_b, old_probs)

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )

import torch


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=0.00067, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os
import numpy as np
import random
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9286
MAX_EP = 3000
MAX_EP_STEP = 200

import  pandas  as pd
df=pd.read_excel('../data/Price_CAISO.xlsx')
from scipy.io import loadmat 
m = loadmat("../data/testingdata.mat")
import numpy as np
out1=np.concatenate((m['out1'],m['out1']),axis=1)
for i in range(5):
    out1=np.concatenate((out1,m['out1']),axis=1)
    out2=np.concatenate((m['out2'],m['out2']),axis=1)
for i in range(5):
    out2=np.concatenate((out2,m['out2']),axis=1)
out3=np.concatenate((m['out3'],m['out3']),axis=1)
for i in range(5):
    out3=np.concatenate((out3,m['out3']),axis=1)
out1=out1.squeeze().astype('int')
out2=out2.squeeze().astype('int')
out3=out3.squeeze().astype('int')
mixed_price=df['Unnamed: 4'].values
ISO_eprice=np.zeros((4000,1))
for i in range(1,1001):
    if mixed_price[9*i-6]>1 and mixed_price[9*i-6]<100:
        ISO_eprice[15*i-14:15*i]=mixed_price[9*i-6]
    elif mixed_price[9*i-6]<100:
        ISO_eprice[15*i-14:15*i]=1
    else:
        ISO_eprice[15*i-14:15*i]=100
import operator
beta1=[-1,-4,-25]
beta2=[6,15,100]
deadline=[6,24,144] #Unit: 5
theta1=0.1
theta2=0.9


max_charging_rate=5 #Unit: 20 KWh
price_upper_bound=int(np.max([-beta2[0]/beta1[0],-beta2[1]/beta1[1],-beta2[2]/beta1[2]]))
#print(price_upper_bound)
N_A=max_charging_rate+price_upper_bound
N_S=8
ISO_eprice=ISO_eprice.astype('float64')
eprice_mean=np.mean(ISO_eprice)

def env(action,residual_demand,iternum):
    if action[1]>residual_demand.shape[0]:
        action[1]=residual_demand.shape[0]

    ##########################Charging Station Start to Charge##########################################
    if residual_demand.shape[0]>0.5:
        #return reward,residual_demand,torch.tensor([0,0,0,0,0])
        least=residual_demand[:,1]-residual_demand[:,0]
        order=[operator.itemgetter(0)(t)-1 for t in sorted(enumerate(least,1), key=operator.itemgetter(1), reverse=True)]
        residual_demand[order[:action[1]],0]=residual_demand[order[:action[1]],0]-1

        residual_demand[:,1]=residual_demand[:,1]-1
    ######################EV Admission##############################################################
    reward=0
    for i in range(out1[iternum]):
        dem=beta1[0]*action[0]+beta2[0]
        if dem<0:
            dem=0
        reward+=dem*action[0]
        residual_demand=demand_update(residual_demand,np.array([dem,deadline[0]]).reshape((1,2)))
    for i in range(out2[iternum]):
        dem=beta1[1]*action[0]+beta2[1]
        if dem<0:
            dem=0
        reward+=dem*action[0]
        residual_demand=demand_update(residual_demand,np.array([dem,deadline[1]]).reshape((1,2)))
    for i in range(out3[iternum]):
        dem=beta1[2]*action[0]+beta2[2]
        if dem<0:
            dem=0
        reward+=dem*action[0]
        residual_demand=demand_update(residual_demand,np.array([dem,deadline[2]]).reshape((1,2)))
    
    if residual_demand.shape[0]<0.5:
        return reward,residual_demand,torch.tensor([0,action[1],0,0,ISO_eprice[iternum+1],out1[iternum+1],out2[iternum+1],out3[iternum+1]])
    #######################Departure#################################################################
    residual_demand_=[]
    for i in range(residual_demand.shape[0]):
        if residual_demand[i,1]>0.5 and residual_demand[i,0]>0.5:
            residual_demand_.append(residual_demand[i,:])
    residual_demand=np.array(residual_demand_)
	######################Caculate Reward and Features##############################################
    f1=reward
    f2=action[1]
    #print(f2)
    try:
        reward_output=reward_output-action[1]*ISO_eprice[iternum]
    except:
        reward_output=reward-action[1]*ISO_eprice[iternum]
    f3=0
    f4=0
    for i in range(residual_demand.shape[0]):
        f3=f3-residual_demand[i,0]*(np.max(residual_demand[:,1])-residual_demand[i,1])*theta1
        f4=f4-residual_demand[i,0]*np.power(theta2,residual_demand[i,1])
    return reward_output, residual_demand, torch.tensor([f1,f2,max(f3,-20),max(f4,-20),5*ISO_eprice[iternum+1]/eprice_mean,out1[iternum+1],out2[iternum+1],out3[iternum+1]])

def demand_update(current,new):
    #print(residual_demand)
    if current.shape[0]<0.5:
        output=new
    else:
        output=np.concatenate((current,new),axis=0)
    return output



class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

        #torch.nn.init.xavier_uniform_(self.pi1)
        #torch.nn.init.xavier_uniform_(self.pi2)
        #torch.nn.init.xavier_uniform_(self.v1)
        #torch.nn.init.xavier_uniform_(self.v2)

    def forward(self, x):
        #print(x.shape)
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        #print(logits[0][0][:max_charging_rate])
        #print(logits[0][0][max_charging_rate:])
        prob1 = F.softmax(logits[0][0][:max_charging_rate], dim=-1).data
        prob2 = F.softmax(logits[0][0][max_charging_rate:], dim=-1).data
        m1 = self.distribution(prob1)
        m2=self.distribution(prob2)
        a1=m1.sample().numpy()
        a2=m2.sample().numpy()
        #print(a1)
        #print(a2)
        return np.array([a1,a2])

    def loss_func(self, s, a, v_t, old_probs):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        #print(logits[:,:max_charging_rate])
        #print(logits[:,max_charging_rate:])

        prob1 = F.softmax(logits[:,:max_charging_rate], dim=-1).data
        prob2 = F.softmax(logits[:,max_charging_rate:], dim=-1).data
        m1 = self.distribution(prob1)
        m2=self.distribution(prob2)
        #print(a.shape)
        #print(a[:,0])
        #print(a[:,1])
        exp_v = m1.log_prob(a[:,0])*m2.log_prob(a[:,1])* td.detach().squeeze()
        a_loss = -exp_v

        # Add entropy term to encourage exploration
        entropy = -((F.log_softmax(logits, dim=-1) * F.softmax(logits, dim=-1)).sum(dim=1))
        entropy_loss = -0.01 * entropy.mean()  # You can adjust the entropy coefficient (e.g., 0.01)

        # Calculate penalty term
        kl_divergence = (old_probs * (old_probs.log() - F.log_softmax(logits, dim=-1))).sum(dim=1)
        penalty = 0.9 * kl_divergence.mean()

        total_loss = (c_loss + a_loss + entropy_loss + penalty).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = env

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            ################################
            #       Initial State          #
            ################################
            a=np.array([100,0])
            s=torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape((1,N_S)).unsqueeze(0)
            real_state=np.array([])
            #########################
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                a = self.lnet.choose_action(s)
                r, real_state_, s_= self.env(a,real_state,t)
                r=np.expand_dims(np.expand_dims(r, 0), 0)
                s_=s_.reshape((1,N_S)).unsqueeze(0).float()
                ep_r += r
                buffer_a.append(np.array(a))
                buffer_s.append(s.squeeze().numpy())
                buffer_r.append(r.squeeze())
                logits, _ = self.lnet.forward(s)
                old_probs = F.softmax(logits.detach(), dim=-1)

                done=False
                if t == MAX_EP_STEP - 1:
                    done = True
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA, old_probs)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    
                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                real_state=real_state_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001126, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # unparallel training
    #workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(1)]
    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
