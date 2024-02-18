import heuristic_agent
import configure
import sac
import utils
import enviroment
import torch
import numpy as np
import pandas as pd
import datahandler
from tqdm import tqdm
from rich import console,progress
import gym
#import envpool
import ray
import time
ray.init()


data=pd.read_hdf('traindata.h5')
dataset=datahandler.requesthandler(data)
datahandler.set_value('data',data)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# alos=[1,3,5,10]
# per=[0.05,0.1,0.2]
# indexing=[(60,120),(500,560)]
alos=[0,3,5]
per=[0.05,0.1,0.2]
indexing=[(500,560)]
# alos=[0]
# per=[0.05]
# indexing=[(60,120)]
def eval(epochs,name,cost_rate=0.452,init_update=0,load=-1,device='cpu',TRAINING_P=0):
    
    writter=SummaryWriter(log_dir='revplogs/re_high_cost_2_FIXTra_trEP'+str(TRAINING_P)+''+name+'/epo'+str(epochs))
    data_=pd.read_hdf('test_data/reqdata.h5')
    qhandle=torch.load('test_data/Qhandle.pt')
    timerec=torch.load('test_data/timerec.pt')
    Zdata=torch.load('test_data/Zdata.pt')
    history_data=pd.read_hdf('test_data/new_traindata.h5')
    #zdata=
    #Qdata=
    data_id=[]
    data_id.append(ray.put(data_))
    data_id.append(ray.put(qhandle))
    data_id.append(ray.put(timerec))
    data_id.append(ray.put(Zdata))
    
    update_times=init_update*5
    progressId={}
    arg_list=[]
    s_e_handle=[]
    arghandle=[]
    for i in range(len(alos)):
        for j in range(len(per)):
            for (start,end) in indexing:
                arghandle.append(''+str(alos[i])+' '+str(per[j])+' '+str(start)+' '+str(end))
                arg_list.append((alos[i],per[j],start,end,data_id,cost_rate,device))
    
    env=gym.make('CarPool-v2',device=device,args_list=arg_list,history_data=history_data,agent_nums=np.array([1],dtype=np.float32))
    

    finished=False
    step=0
    # if name=='LinTS':
    #     agents=heuristic_agent.multimultiAgents([
    #         heuristic_agent.LinTSMultiArmbanditAgent.remote(arms=9,dimension=5) for _ in range(len(arg_list))
    #         ],arghandle)
    if name=='LinTS':
        agents=heuristic_agent.Env2multiAgents([
            heuristic_agent.LinTSMultiArmbanditAgentIn2.remote(arms=9,dimension=5) for _ in range(len(arg_list))
            ],arghandle)
    if name=='Tom':
        agents=heuristic_agent.multiAgents([
        heuristic_agent.multiArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
    ],arghandle)
    if name=='greedy':
        agents=heuristic_agent.multiAgents([
            #heuristic_agent.multiArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
            heuristic_agent.mutgreedAgent.remote() for _ in range(len(arg_list))
            #heuristic_agent.estimatemutiArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
            #heuristic_agent.multiQuoMultiArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
            #heuristic_agent.multiTomsArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
            #heuristic_agent.MulLinTSMultiArmbanditAgent.remote(arms=9,dimension=5) for _ in range(len(arg_list))
            #heuristic_agent.LinTSMultiArmbanditAgent.remote(arms=9,dimension=5) for _ in range(len(arg_list))
            #heuristic_agent.randomAgent.remote() for _ in range(len(arg_list))
        ],arghandle
        )
    if name=='ucb':
        agents=heuristic_agent.multiAgents([
        heuristic_agent.multiQuoMultiArmBanditAgent.remote(arms=9) for _ in range(len(arg_list))
    ],arghandle)
    if name=='D4QN':
        agents=heuristic_agent.multiAgents([
        heuristic_agent.DistributionalD3QNUCBAgentWithPER.remote(arms=9) for _ in range(len(arg_list))
    ],arghandle)
    if name=='D4VQN':
        agents=heuristic_agent.multiAgents([
        heuristic_agent.DistributionalD3QNVUCBAgentWithPER.remote(arms=9) for _ in range(len(arg_list))
    ],arghandle)
    if name=='D4EQN':
        agents=heuristic_agent.multiAgents([
        heuristic_agent.DistributionalD3QNEAgentWithPER.remote(arms=9) for _ in range(len(arg_list))
    ],arghandle)
    if name=='e2eSAC':
        agents=heuristic_agent.E2EmultiAgents([
        heuristic_agent.End2EndSACAgent.remote(max_act=60) for _ in range(len(arg_list))
    ],arghandle)
    if name=='Hyper':
        agents=heuristic_agent.HyperMultiAgent([
            heuristic_agent.HyperAgent.remote() for _ in range(len(arg_list))
        ],arghandle)
    if name=='EPSHyper':
        agents=heuristic_agent.HyperMultiAgent([
            heuristic_agent.HyperEPSAgent.remote() for _ in range(len(arg_list))
        ],arghandle)
    #info={'driver_nums':np.zeros((len(arg_list),11),dtype=np.float32)}
    with tqdm() as pbar:
        training_counter=0
        while training_counter<TRAINING_P:
            obs,info=env.reset()
            while not finished:
                training_counter+=1
                step+=1
                t=time.perf_counter()
                # if len(obs)!=4:
                #     obs=zip(*obs)
                act,actinfo=agents.acts(*obs,info)
                print(time.perf_counter()-t)
                obs_next,rewards,done,info=env.step((act,None))
                pbar.total=info['all'][0]
                pbar.n=info['done'][0]
                pbar.update(0)
                if type(act)!=np.ndarray:
                    printact=act.cpu().numpy()
                else:
                    printact=act
                #pbar.write(np.array2string(printact.T))
                if any(done):
                    finished=True                    
                Tr=[]
                obs_context=agents.obs2Context(*obs,info)
                next_obs_context=agents.obs2Context(*obs_next,info)
                for i in range(act.shape[0]):
                    Tr.append((obs_context[i],act[i],next_obs_context[i],rewards[i],done[i]))
                agents.updating(Tr)
                obs=obs_next
                #logging(info,step,writter,arghandle,obs[0][0],obs[1][0],act)
        if TRAINING_P!=0:
            env=gym.make('CarPool-v2',device=device,args_list=arg_list,history_data=history_data,agent_nums=np.array([1],dtype=np.float32))
        obs,info=env.reset()
        #pbar.total=info['all'][0]
        #pbar.n=0
        finished=False
        while not finished:
            training_counter+=1
            step+=1
            #t=time.perf_counter()
            # if len(obs)!=4:
            #     obs=zip(*obs)
            act,actinfo=agents.acts(*obs,info)
            #print(time.perf_counter()-t)
            obs_next,rewards,done,info=env.step((act,None))
            pbar.total=info['all'][0]
            pbar.n=info['done'][0]
            pbar.update(0)
            if type(act)!=np.ndarray:
                printact=act.cpu().numpy()
            else:
                printact=act
            #pbar.write(np.array2string(printact.T))
            if any(done):
                finished=True                    
            Tr=[]
            obs_context=agents.obs2Context(*obs,info)
            next_obs_context=agents.obs2Context(*obs_next,info)
            for i in range(act.shape[0]):
                Tr.append((obs_context[i],act[i],next_obs_context[i],rewards[i],done[i]))
            agents.updating(Tr)
            obs=obs_next
            logging(info,step,writter,arghandle,obs[0][0],obs[1][0],act,actinfo=actinfo)
        
            #env=gym.make('CarPool-v0',driver_nums=dri_nums,cost_rate=cost_rate,device=device,envid=0) 
            #env=gym.vector.make('CarPool-v0',num_envs=8,asynchronous=False,dri_nums=dri_nums,cost_rate=cost_rate,device=device,envid=0)
            

def logging(info,step,writter,arglist,Q,Q_dist,act,numagents=len(configure.cost_rate),actinfo=None):
    cum_rewards=info['cum_reward']
    cum_reward_dict={}
    for i in range(len(cum_rewards)):
        cum_reward_dict['cum_reward'+arglist[i]]=cum_rewards[i]
    writter.add_scalars('cum_reward',cum_reward_dict,step)
    drivers=info['driver_nums']
    driver_dict={}
    for i in range(len(drivers)):
        sumdri=np.sum(drivers[i])
        driver_dict['driver_per'+arglist[i]]=0
        for j in range(numagents):
            driver_dict['driver_per'+arglist[i]]+=drivers[i][j]/sumdri
    writter.add_scalars('driver_per',driver_dict,step)
    getted_req=info['getted_req']
    getted_req_dict={}
    reward_per_req_dict={}
    for i in range(len(getted_req)):
        getted_req_dict['getted_req'+arglist[i]]=getted_req[i]/step
        reward_per_req_dict['reward_per_req'+arglist[i]]=0 if getted_req[i]==0 else cum_rewards[i]/getted_req[i]
    writter.add_scalars('getted_req',getted_req_dict,step)
    writter.add_scalars('reward_per_req',reward_per_req_dict,step)
    # Q_dist_dict={}
    # for i in range(len(configure.Qquan)):
    #     Q_dist_dict.update({'Q_dist'+str(configure.Qquan[i]):Q_dist[0][i].item()})
    # Q_dist_dict.update({'Q_value':Q.item()})
    # writter.add_scalars('Q_value',Q_dist_dict,step)
    act_dist={}
    #print(Q_dist)
    for i in range(len(getted_req)):
        for j in range(numagents):
            act_dist.update({'act'+arglist[i]+'_'+str(j):np.round((act[i][j].item())/Q_dist[j],1,)})
    writter.add_scalars('act',act_dist,step) 
    if actinfo!=None:
        actinfo_dict={}
        for i in range(len(getted_req)):
            actinfo_dict.update({'actinfo'+arglist[i]:actinfo[i]})
        writter.add_scalars('actinfo',actinfo_dict,step)
if __name__=='__main__':
    # for i in range(20):
    #     eval(i,'LinTS')
    #for i in range(10):
    #    eval(i,'greedy')
    # for i in range(20):
    #     eval(i,'D4QN')
    # for i in range(20):
    #     eval(i,'D4VQN')
    # for i in range(20):
    #     eval(i,'D4EQN')
    # for i in range(10):
    #     eval(i,'EPSHyper',TRAINING_P=0)
    for i in range(10):
        eval(i,'EPSHyper',TRAINING_P=1)
    for i in range(10):
        eval(i,'EPSHyper',TRAINING_P=2)
    
    # for i in range(20):
    #     eval(i,'Hyper',TRAINING_P=2)
    
    # for i in range(10):
    #     eval(i,'e2eSAC',TRAINING_P=0)
    # for i in range(10):
    #     eval(i,'e2eSAC',TRAINING_P=1)
    # for i in range(10):
    #     eval(i,'e2eSAC',TRAINING_P=2)