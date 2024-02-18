import numpy as np
import ray
import torch
import configure
import torch.nn as nn
from collections import namedtuple
import copy
import torch.optim as optim
@ray.remote(num_cpus=1)
class greedAgent():
    def __init__(self):
        self.reward_dict={}
        self.action_list=[]
        self.get=0
        pass
    def act(self,Q,Q_dist,quos,cost,info=None):
        return torch.tensor([-1.])
    def add_reward(self,minutes,reward,act,ifget):
        if minutes not in self.reward_dict.keys():
            self.reward_dict[minutes]=reward
        else:
            self.reward_dict[minutes]+=reward
        self.action_list.append(act)
        if ifget:
            self.get+=1
    def updating(self,trans):
        pass
@ray.remote(num_cpus=1)
class randomAgent():
    def __init__(self):
        self.reward_dict={}
        self.action_list=[]
        self.get=0
        pass
    def act(self,Q,Q_dist,quos,cost,info=None):
        rand=np.random.rand()
        rand=(rand-0.5)*2
        rand=torch.tensor([rand],dtype=torch.float32)
        return rand
    def add_reward(self,minutes,reward,act,ifget):
        if minutes not in self.reward_dict.keys():
            self.reward_dict[minutes]=reward
        else:
            self.reward_dict[minutes]+=reward
        self.action_list.append(act)
        if ifget:
            self.get+=1
    def updating(self,trans):
        pass
@ray.remote(num_cpus=1)
class mutgreedAgent():
    def __init__(self):
        self.reward_dict={}
        self.action_list=[]
        self.get=0
        pass
    def act(self,Q,Q_dist,quos,cost,info=None):
        return torch.tensor([-1.])
    #   return torch.tensor([-1.,-1.])
    def add_reward(self,minutes,reward,act,ifget):
        if minutes not in self.reward_dict.keys():
            self.reward_dict[minutes]=reward
        else:
            self.reward_dict[minutes]+=reward
        self.action_list.append(act)
        if ifget:
            self.get+=1
    def updating(self,trans):
        pass
@ray.remote(num_cpus=1)
class multiArmBanditAgent():
    def __init__(self,arms):
        self.reward_dict={}
        self.arms=arms
        self.arms_success=[0 for i in range(arms)]
        self.arms_success=np.array(self.arms_success)
        self.arms_fail=[0 for i in range(arms)]
        self.arms_fail=np.array(self.arms_fail)
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
    
    def get_probs(self):
        probs=[]
        for i in range(self.arms):
            prob=np.random.beta(self.arms_success[i]+1,self.arms_fail[i]+1)
            probs.append(prob)
        return np.array(probs)
    def act(self,Q,Q_dist,quos,cost,info=None):
        probs=self.get_probs()
        probs=torch.from_numpy(probs)
        temp=[]
        for q in range(len(self.arms_quantile)):
            temp.append(self.act_to_quo(quos,self.arms_quantile[q]))
        act_quos=torch.tensor(temp,dtype=torch.float32)
        vls=act_quos
        vls=probs*vls
        act=np.argmax(vls)
        return torch.tensor([self.arms_quantile[act].item()])
    def log(self):
        pass
    def act_to_quo(self,quos,act):
                # less=act<0
        # bigger=act>=0
        # q=less*((quos[1]-quos[0])*act+quos[1])
        # q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        # return q
        # quos:[b,5]
        # segmental linear interpolation [-1,-0.5,0,0.5,1]
        q=0
        
        if -1 <= act and act < -0.5:
            q=quos[0]+(quos[1]-quos[0])*2*(act+1)
        elif -0.5 <= act and act < 0:
            q=quos[1]+(quos[2]-quos[1])*2*(act+0.5)
        elif 0 <= act and act < 0.5:
            q=quos[2]+(quos[3]-quos[2])*2*(act)
        elif 0.5 <= act and act <= 1:
            q=quos[3]+(quos[4]-quos[3])*2*(act-0.5)
        return torch.tensor([q],dtype=torch.float32)
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        arms=torch.argwhere(acts==self.arms_quantile)
        if rewards[1]==0:
            self.arms_success[arms]+=1
        else:
            self.arms_fail[arms]+=1
            
@ray.remote(num_cpus=1)
class multiTomsArmBanditAgent():
    def __init__(self,arms,num_agents=2):
        self.reward_dict={}
        self.arms=arms
        self.arms_success=[0 for i in range(arms)]
        self.arms_success=[np.array(self.arms_success) for i in range(num_agents)]
        self.arms_fail=[0 for i in range(arms)]
        self.arms_fail=[np.array(self.arms_fail) for i in range(num_agents)]
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        self.cost_rate=configure.cost_rate
        self.num_agents=num_agents
    def get_probs(self,index):
        probs=[]
        for i in range(self.arms):
            prob=np.random.beta(self.arms_success[index][i]+1,self.arms_fail[index][i]+1)
            probs.append(prob)
        return np.array(probs)
    def act(self,Q,Q_dist,quos,cost,info=None):
        ans=[]
        for i in range(self.num_agents):
            probs=self.get_probs(i)
            probs=torch.from_numpy(probs)
            temp=[]
            for q in range(len(self.arms_quantile)):
                temp.append(self.act_to_quo(quos,self.arms_quantile[q]))
            act_quos=torch.tensor(temp,dtype=torch.float32)
            vls=act_quos-cost*self.cost_rate[i]
            vls=probs*vls
            act=np.argmax(vls)
            ans.append(self.arms_quantile[act])
        return torch.tensor(ans)
    def log(self):
        pass
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        for i in range(acts.shape[0]):
            arms=torch.argwhere(acts[i]==self.arms_quantile)
            if rewards[1]==i:
                self.arms_success[i][arms]+=1
            else:
                self.arms_fail[i][arms]+=1
                
@ray.remote(num_cpus=1)
class multiQuoMultiArmBanditAgent():
    def __init__(self,arms,agents_num=len(configure.cost_rate)):
        self.arms=arms
        self.agents_num=agents_num
        self.counts=[torch.zeros(arms,dtype=torch.float32) for _ in range(agents_num)]
        self.values=[torch.zeros(arms,dtype=torch.float32) for _ in range(agents_num)]
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        self.cost_rate=configure.cost_rate
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
        # quos:[b,5]
        # segmental linear interpolation [-1,-0.5,0,0.5,1]
        # q=0
        
        # if -1 <= act and act < -0.5:
        #     q=quos[0]+(quos[1]-quos[0])*2*(act+1)
        # elif -0.5 <= act and act < 0:
        #     q=quos[1]+(quos[2]-quos[1])*2*(act+0.5)
        # elif 0 <= act and act < 0.5:
        #     q=quos[2]+(quos[3]-quos[2])*2*(act)
        # elif 0.5 <= act and act <= 1:
        #     q=quos[3]+(quos[4]-quos[3])*2*(act-0.5)
        # return torch.tensor([q],dtype=torch.float32)
    def act(self,Q,Q_dist,quos,cost,info=None):
        temp=[]
        for q in range(len(self.arms_quantile)):
            temp.append(self.act_to_quo(quos,self.arms_quantile[q]))
        act_quos=torch.tensor(temp,dtype=torch.float32)
        act=[]
        for i in range(self.agents_num):
            if torch.any(self.counts[i])==0:
                act.append(self.arms_quantile[torch.argwhere(self.counts[i]==0)[0]])
            else:
                ucbs=self.get_ucb(i)
                values=act_quos-cost*self.cost_rate[i]
                a=torch.argmax(ucbs*values)
                act.append(self.arms_quantile[a])
        return torch.tensor(act)
    def get_ucb(self,i):
        total=self.counts[i].sum()
        constant=torch.broadcast_to(total,self.counts[i].shape)
        bonus=torch.sqrt(torch.log(constant)/self.counts[i])
        return self.values[i]+bonus
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        
        for i in range(acts.shape[0]):

            arms=torch.argwhere(acts[i]==self.arms_quantile)
            self.counts[i][arms]=self.counts[i][torch.argwhere(acts[i]==self.arms_quantile)]+1
            if rewards[1]==i:
                self.values[i][arms]=(self.counts[i][arms]/(1+self.counts[i][arms]))*self.values[i][arms]+(1/(1+self.counts[i][arms]))*1
            else:
                self.values[i][arms]=(self.counts[i][arms]/(1+self.counts[i][arms]))*self.values[i][arms]+(1/(1+self.counts[i][arms]))*0
@ray.remote(num_cpus=1)
class estimatemutiArmBanditAgent():
    def __init__(self,arms):
        self.reward_dict={}
        self.arms=arms
        self.arms_success=[0 for i in range(arms)]
        self.arms_success=np.array(self.arms_success)
        self.arms_fail=[0 for i in range(arms)]
        self.arms_fail=np.array(self.arms_fail)
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
    def get_probs(self):
        probs=[]
        for i in range(self.arms):
            prob=np.random.beta(self.arms_success[i]+1,self.arms_fail[i]+1)
            probs.append(prob)
        return np.array(probs)
    def act(self,Q,Q_dist,quos,cost,info=None):
        if self.policy(Q_dist,Q):
            probs=self.get_probs()
            probs=torch.from_numpy(probs)
            temp=[]
            for q in range(len(self.arms_quantile)):
                temp.append(self.act_to_quo(quos,self.arms_quantile[q]))
            act_quos=torch.tensor(temp,dtype=torch.float32)
            vls=act_quos
            vls=probs*vls
            act=np.argmax(vls)
            act=self.arms_quantile[act]
            return act
        else:
            return self.arms_quantile[np.random.randint(0,self.arms)]
    def log(self,arg_str):
        return self.get_probs()
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        arms=torch.argwhere(acts==self.arms_quantile)
        if rewards>0:
            self.arms_success[arms]+=1
        else:
            self.arms_fail[arms]+=1
        
    def est_toler(self):
        probs=self.get_probs()
        toler=probs/np.max(probs)
        return toler,np.max(probs)
    def policy(self,Q_dist,Q):
        Q_dist=Q_dist[0]
        toler,estm=self.est_toler()
        #print(Q_dist)
        #print(toler)
        #print(estm)
        selected_rate=np.sum(toler)/self.arms
        allow=min(selected_rate,1)
        allow=1-allow
        #print(allow)
        #print(selected_rate)
        #print(Q)
        rank=0
        while not rank==len(Q_dist)-1 and not (Q>Q_dist[rank] and Q<=Q_dist[rank+1]) :
            rank+=1
        rankquan=0
        if rank==len(Q_dist)-1:
            rankquan=1
        else:
            rankquan=configure.Qquan[rank]
        #print(rankquan)
        if rankquan>allow:
            return True
        return False

@ray.remote(num_cpus=1)
class LinTSMultiArmbanditAgent():
    def __init__(self,arms,dimension,agents_num=[0],delta=0.5,epsilon=1/np.log(20000),R=0.01):
        self.v=[R*np.sqrt(24/epsilon*dimension*np.log(1/delta)) for i in range(arms)]
        self.B=[np.identity(dimension) for i in range(arms)]
        self.mu_hat=[np.zeros((dimension,1)) for i in range(arms)]
        self.f=[np.zeros((dimension,1)) for i in range(arms)]
        self.arms=arms
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        self.last_arm=None
        self.last_context=None
        self.agents_num=agents_num
    def sampled(self,context,index):
        v=self.v[index]
        Bm=self.B[index]
        mu_hat=self.mu_hat[index]
        f=self.f[index]
        param1=np.matmul(context.T,mu_hat)
        param2=v**2*np.matmul(np.matmul(context.T,np.linalg.inv(Bm)),context)
        return np.random.normal(param1,param2)
    def act(self,Q,Q_dist,quos,cost,info=None):
        assert info != None
        context=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(-1,1)
        temp=np.zeros((self.arms))
        for i in range(self.arms):
            temp[i]=self.sampled(context,i)
        self.last_arm=np.argmax(temp)
        self.last_context=context
        return self.arms_quantile[self.last_arm].unsqueeze(0)
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        rew=torch.tensor([0.])
        if rewards[1] in self.agents_num:
            rew=rewards[0]
        arm=self.last_arm
        context=self.last_context
        self.B[arm]+=np.matmul(context,context.T)
        self.f[arm]+=context*rew.numpy()
        self.mu_hat[arm]=np.matmul(np.linalg.inv(self.B[arm]),self.f[arm])
        
        
@ray.remote(num_cpus=1)
class MulLinTSMultiArmbanditAgent():
    class subLinTSMultiArmbanditAgent():
        def __init__(self,arms,dimension,agents_num=[0],delta=0.5,epsilon=1/np.log(20000),R=0.01):
            self.v=[R*np.sqrt(24/epsilon*dimension*np.log(1/delta)) for i in range(arms)]
            self.B=[np.identity(dimension) for i in range(arms)]
            self.mu_hat=[np.zeros((dimension,1)) for i in range(arms)]
            self.f=[np.zeros((dimension,1)) for i in range(arms)]
            self.arms=arms
            self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
            self.arms_quantile=np.array(self.arms_quantile)
            self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
            self.arms_quantile=self.arms_quantile*2
            self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
            self.last_arm=None
            self.last_context=None
            self.agents_num=agents_num
        def sampled(self,context,index):
            v=self.v[index]
            Bm=self.B[index]
            mu_hat=self.mu_hat[index]
            f=self.f[index]
            param1=np.matmul(context.T,mu_hat)
            param2=v**2*np.matmul(np.matmul(context.T,np.linalg.inv(Bm)),context)
            return np.random.normal(param1,param2)
        def act(self,Q,Q_dist,quos,cost,info=None):
            assert info != None
            context=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(-1,1)
            temp=np.zeros((self.arms))
            for i in range(self.arms):
                temp[i]=self.sampled(context,i)
            self.last_arm=np.argmax(temp)
            self.last_context=context
            return self.arms_quantile[self.last_arm].unsqueeze(0)
        def updating(self,transition):
            obs,acts,obs_next,rewards,done=transition
            rew=0
            if rewards[1] in self.agents_num:
                rew=rewards[0].numpy()
            arm=self.last_arm
            context=self.last_context
            self.B[arm]+=np.matmul(context,context.T)
            self.f[arm]+=context*rew
            self.mu_hat[arm]=np.matmul(np.linalg.inv(self.B[arm]),self.f[arm])
    def __init__(self,arms,dimension,agents_num=[0,1],delta=0.5,epsilon=1/np.log(20000),R=0.01):
        self.single=[self.subLinTSMultiArmbanditAgent(arms,dimension,[agents_num[i]],delta,epsilon,R) for i in range(2)]
        self.agents_num=agents_num
    def act(self,Q,Q_dist,quos,cost,info=None):
        acts=[]
        for i in range(len(self.agents_num)):
            acts.append(self.single[i].act(Q,Q_dist,quos,cost,info[i:i+1]))
        return torch.cat(acts,dim=0)
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        for i in range(len(self.agents_num)):
            self.single[i].updating((obs,acts[i],obs_next,rewards,done))
            
class multiAgents():
    def __init__(self,agents,arg_str):
        self.agents=agents
        self.arg_str=arg_str
    def acts(self,Q,Q_dist,quos,cost,info=None):
        handle=[]
        for i,agent in enumerate(self.agents):
            dri_nums=info['driver_nums'][i]
            handle.append(agent.act.remote(Q[i],Q_dist[i],quos[i],cost[i]*configure.cost_rate[0],dri_nums[0:len(configure.cost_rate)]))
        actions=ray.get(handle)
        return torch.stack(actions)
    def updating(self,transitions):
        for i,agent in enumerate(self.agents):
            agent.updating.remote(transitions[i])
    def log(self,writter):
        for i,agent in enumerate(self.agents):
            agent.log.remote(self.arg_str[i],writter)
    def obs2Context(self,Q,Q_dist,quos,cost,info=None):
        ctxs=[]
        for i in range(len(self.agents)):
            dri_nums=info['driver_nums'][i]
            context=np.array([*quos[i].numpy(),*cost[i].unsqueeze(0).numpy()*configure.cost_rate[0],dri_nums[i]]).reshape(1,-1)
            ctxs.append(torch.from_numpy(context).float())
        ctx=torch.cat(ctxs,dim=0)
        return ctx
            

            
@ray.remote(num_cpus=1)
class DistributionalD3QNUCBAgent():
    def __init__(self,arms,agents_num=[0],update_inter=10,quantiles=10,lr=0.005,epsilon=0.1,tau=10,buffer_size=1000,batch_size=16):
        # distributional double dueling deep Q network
        class replayBuffer:
            def __init__(self,batch_size,buffer_size):
                self.buffer_size=buffer_size
                self.batch_size=batch_size
                self.buffer=[]
                self.index=0
            def add(self,transition):
                if len(self.buffer)<self.buffer_size:
                    self.buffer.append(transition)
                else:
                    self.buffer[self.index]=transition
                    self.index=(self.index+1)%self.buffer_size
            def sample(self):
                ind=np.random.choice(len(self.buffer),self.batch_size)
                data=[self.buffer[i] for i in ind]
                transed=list(zip(*data))
                for i in range(len(transed)):
                    transed[i]=torch.stack(transed[i],dim=0)
                return transed
            def __len__(self):
                return len(self.buffer)

        class NetModule(nn.Module):
            def __init__(self,context_len,act_space,quantiles,dim1=32,dim2=64,dim3=32,gamma=0.9):
                super(NetModule,self).__init__()
                self.quantiles=quantiles
                self.act_space=act_space
                self.context_len=context_len
                self.gamma=gamma
                self.hidden1=nn.Sequential(
                    nn.Linear(context_len,dim1),
                    nn.ReLU(),
                    nn.Linear(dim1,dim2),
                    nn.ReLU()
                )
                self.A1=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,act_space*quantiles)
                )
                self.V1=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,quantiles)
                )
                
                self.hidden2=nn.Sequential(
                    nn.Linear(context_len,dim1),
                    nn.ReLU(),
                    nn.Linear(dim1,dim2),
                    nn.ReLU()
                )
                self.A2=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,act_space*quantiles)
                )
                self.V2=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,quantiles)
                )
                self.qvs=torch.linspace(0.0,1.0-1/(quantiles),quantiles)+1/(2*quantiles)
                
                def loss(quan,target,pred,delta=0.5):
                    u=target-pred
                    weights=torch.abs(quan-(u<0).float()).detach()
                    huber=nn.functional.huber_loss(pred,target,reduction='none',delta=delta)
                    return torch.mean(weights*huber)
                
                self.loss=loss
                
            def forward(self,context):
                context=self.hidden1(context)
                v=self.V1(context) # batch,quantiles
                a=self.A1(context) # batch,act_space*quantiles
                a=a.view(-1,self.act_space,self.quantiles) # batch,act_space,quantiles
                v=v.view(-1,1,self.quantiles) # batch,1,quantiles
                q=torch.add(v,a)-torch.mean(a,dim=1,keepdim=True) # batch,act_space,quantiles
                return q
            
            def target_foward(self,context):
                context=self.hidden2(context)
                v=self.V2(context) # batch,quantiles
                a=self.A2(context) # batch,act_space*quantiles
                a=a.view(-1,self.act_space,self.quantiles) # batch,act_space,quantiles
                v=v.view(-1,1,self.quantiles) # batch,1,quantiles
                q=torch.add(v,a)-torch.mean(a,dim=1,keepdim=True) # batch,act_space,quantiles
                return q
            
            def update_target(self,tau=0.2):
                for target,param in zip(self.hidden2.parameters(),self.hidden1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                for target,param in zip(self.A2.parameters(),self.A1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                for target,param in zip(self.V2.parameters(),self.V1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                
            def act(self,context,stra='epsilon_greedy'):
                q=self.forward(context)
                if stra=='epsilon_greedy':
                    if np.random.rand()<0.1:
                        return torch.randint(0,self.act_space,(1,))
                    else:
                        return torch.argmax(torch.mean(q,dim=2),dim=1)
                    
            def tderror(self,transition):
                #print(transition)
                state,action,next_state,reward,done=transition
                
                new_value=self.forward(next_state)
                new_q=new_value.mean(dim=2)
                new_act=torch.argmax(new_q,dim=1)
                # print(new_act.shape) #[batch]
                # print(self.target_foward(next_state).shape) [batch,act_space,quantiles]
                # print(new_q.shape)
                # print(self.target_foward(next_state)[:,new_act,:].shape)
                # print((1-done.float()).shape)
                tfrd=self.target_foward(next_state)
                next_z=torch.stack([tfrd[i,new_act[i],:] for i in range(len(new_act))],dim=0)
                #print(next_z.shape,(1-done.float()).shape,reward.shape)
                target_z=(self.gamma*next_z*(1-done.float())+reward.unsqueeze(1)).detach()
                frd=self.forward(state)
                
                z_now=torch.stack([frd[i,action[i],:] for i in range(len(action))],dim=0)
                #print(z_now.shape,target_z.shape,self.qvs.shape)
                z_now=z_now[:,0,:]
                _tderror=self.loss(self.qvs,target_z,z_now)
                return _tderror
        
        self.transition=namedtuple('transition',['state','action','next_state','reward','done'])
        self.epsilon=epsilon
        self.tau=tau
        self.buffer_size=buffer_size
        self.update_inter=update_inter
        self.lr=lr
        self.arms=arms
        self.agents_num=agents_num
        self.batch_size=batch_size
        self.quantiles=quantiles
        self.net=NetModule(context_len=5,act_space=arms,quantiles=quantiles)
        self.replay_buffer=replayBuffer(buffer_size=buffer_size,batch_size=batch_size)
        #self.last_state,self.last_action,self.last_reward,self.last_done=None,None,None,None
        self.train_counter=0
        self.soft_counter=0
        self.opt=torch.optim.Adam(self.net.parameters(),lr=lr)
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
    def act(self,Q,Q_dist,quos,cost,info=None):
        Q,Q_dist = None,None
        assert info != None
        context=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(1,-1)
        state=torch.from_numpy(context).float()
        act=self.net.act(state)
        # if self.last_state is not None:
        #     tran=self.transition(self.last_state,self.last_action,state,self.last_reward,False)
        # self.last_state,self.last_action=context,act
        act=self.arms_quantile[act]
        return act
        
    def updating(self,transition):
        state,action,next_state,reward,done=transition
        reward=reward[0]
        transition=self.transition(state,action,next_state,reward,done)
        #print(transition)
        self.replay_buffer.add(transition)
        self.train_counter+=1
        if self.train_counter%(self.batch_size)==1:
            batch=self.replay_buffer.sample()
            #print(batch)
            loss=self.net.tderror(batch)
            self.net.zero_grad()
            loss.backward()
            self.opt.step()
            self.soft_counter+=1
            if self.soft_counter%self.tau==0:
                self.net.update_target()

@ray.remote(num_cpus=1)
class DistributionalD3QNUCBAgentWithPER():
    def __init__(self,arms,agents_num=[0],update_inter=10,quantiles=10,lr=0.005,epsilon=0.05,tau=10,buffer_size=1000,batch_size=16):
        # distributional double dueling deep Q network
        class replayBuffer():
            def __init__(self,capacity,batch_size,alpha=0.1,beta=0.1,beta_increment_per_sampling=0.001):
                self.e=0.001
                self.capacity=capacity
                self.batch_size=batch_size
                self.alpha=alpha
                self.beta=beta
                self.beta_increment_per_sampling=beta_increment_per_sampling
                
                class Sum_Tree():
                    
                    def __init__(self,capacity):
                        self.tree=np.zeros(2*capacity-1)
                        self.data=np.zeros(capacity,dtype=object)
                        self.entries=0
                        self.write=0
                        self.capacity=capacity
                        self.max_prio=10
                    def _propagate(self, idx, change):
                        parent = (idx - 1) // 2
                        self.tree[parent] += change
                        if parent != 0:
                            self._propagate(parent, change)
                            
                    def _retrieve(self, idx, s):
                        left = 2 * idx + 1
                        right = left + 1
                        if left >= len(self.tree):
                            return idx
                        if s <= self.tree[left]:
                            return self._retrieve(left, s)
                        else:
                            return self._retrieve(right, s - self.tree[left])
                        
                    def _update(self,idx,p):
                        change=p-self.tree[idx]
                        self.tree[idx]=p
                        self._propagate(idx,change)
                        if p>self.max_prio:
                            self.max_prio=p
                    def _add(self, p,data):
                        idx=self.write+self.capacity-1
                        self.data[self.write]=data
                        self._update(idx,p)
                        self.write+=1
                        if self.write>=self.capacity:
                            self.write=0
                        if self.entries<self.capacity:
                            self.entries+=1
                        if p>self.max_prio:
                            self.max_prio=p
                            
                    def _get(self,s):
                        idx=self._retrieve(0,s)
                        dataidx=idx-self.capacity+1
                        return (idx,self.tree[idx],self.data[dataidx])
                    
                    def total(self):
                        return self.tree[0]
                    
                self.tree=Sum_Tree(capacity)
                
            def _get_priority(self,error):
                return (np.abs(error)+self.e)**self.alpha
            
            def add(self,tran,error=None):
                if error is None:
                    p=self.tree.max_prio
                else:
                    p=self._get_priority(error)

                self.tree._add(p,tran)

            def sample(self,batchs=0):
                if batchs==0:
                    batchs=self.batch_size
                sampled=[]
                idxs=[]
                pris=[]
                segment=self.tree.total()/batchs
                
                self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
                for _ in range(batchs):
                    a=segment*_
                    b=segment*(_+1)
                    #print(a,b)
                    s=np.random.uniform(a,b)
                    (idx,p,data)=self.tree._get(s)
                    idxs.append(idx)
                    pris.append(torch.tensor([p],dtype=float))
                    sampled.append(data)
                #print(sampled)
                transed=list(zip(*sampled))
                for i in range(len(transed)):
                    transed[i]=torch.stack(transed[i],dim=0)
                pris=torch.stack(pris,dim=0).float()
                pris=pris/self.tree.total()
                weights=torch.pow(self.tree.entries*pris,-self.beta)
                weights=weights/torch.max(weights,dim=0,keepdim=True)[0]
                #print(transed,idxs,weights)
                return transed,idxs,weights
            
            def update(self,idx,error):
                p=self._get_priority(error)
                self.tree._update(idx,p)

        class NetModule(nn.Module):
            def __init__(self,context_len,act_space,quantiles,dim1=32,dim2=64,dim3=32,gamma=0.99,epsilon=0.05):
                super(NetModule,self).__init__()
                self.quantiles=quantiles
                self.act_space=act_space
                self.context_len=context_len
                self.gamma=gamma
                self.hidden1=nn.Sequential(
                    nn.Linear(context_len,dim1),
                    nn.ReLU(),
                    nn.Linear(dim1,dim2),
                    nn.ReLU()
                )
                self.A1=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,act_space*quantiles)
                )
                self.V1=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,quantiles)
                )
                
                self.hidden2=nn.Sequential(
                    nn.Linear(context_len,dim1),
                    nn.ReLU(),
                    nn.Linear(dim1,dim2),
                    nn.ReLU()
                )
                self.A2=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,act_space*quantiles)
                )
                self.V2=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,quantiles)
                )
                self.qvs=torch.linspace(0.0,1.0-1/(quantiles),quantiles)+1/(2*quantiles)
                
                def loss(quan,target,pred,delta=0.5):
                    u=target-pred
                    weights=torch.abs(quan-(u<0).float()).detach()
                    huber=nn.functional.huber_loss(pred,target,reduction='none',delta=delta)
                    return (weights*huber).mean(dim=1)
                
                self.loss=loss
                self.epsilon=epsilon
            def forward(self,context):
                context=self.hidden1(context)
                v=self.V1(context) # batch,quantiles
                a=self.A1(context) # batch,act_space*quantiles
                a=a.view(-1,self.act_space,self.quantiles) # batch,act_space,quantiles
                a=torch.abs(a)
                a=a.cumsum(dim=2)
                v=v.view(-1,1,self.quantiles)# batch,1,quantiles
                v=torch.abs(v)
                v=v.cumsum(dim=2)
                q=torch.add(v,a)-torch.mean(a,dim=1,keepdim=True) # batch,act_space,quantiles
                return q
            
            def target_foward(self,context):
                context=self.hidden2(context)
                v=self.V2(context) # batch,quantiles
                a=self.A2(context) # batch,act_space*quantiles
                a=a.view(-1,self.act_space,self.quantiles) # batch,act_space,quantiles
                v=v.view(-1,1,self.quantiles) # batch,1,quantiles
                q=torch.add(v,a)-torch.mean(a,dim=1,keepdim=True) # batch,act_space,quantiles
                return q
            
            def update_target(self,tau=0.2):
                for target,param in zip(self.hidden2.parameters(),self.hidden1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                for target,param in zip(self.A2.parameters(),self.A1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                for target,param in zip(self.V2.parameters(),self.V1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                
            def act(self,context,stra='simpleUCB',arms_counter=None,windows=1000):
                q=self.forward(context)
                if stra=='epsilon_greedy':
                    if np.random.rand()<0.1:
                        return torch.randint(0,self.act_space,(1,))
                    else:
                        return torch.argmax(torch.mean(q,dim=2),dim=1)
                if stra=='simpleUCB':
                    assert arms_counter is not None
                    CONSTANT=3
                    arms_counter=torch.sum(arms_counter,dim=0)
                    total=arms_counter.sum()
                    upper=CONSTANT*torch.sqrt(2*torch.log(total)/arms_counter)
                    mean=torch.mean(q,dim=2)
                    upperBound=mean+upper
                    return torch.argmax(upperBound,dim=1)
                if stra=='VarianceUCB':
                    #calculate v sub gaussian
                    mean=torch.mean(q,dim=2)
                    arms_counter=torch.sum(arms_counter,dim=0)
                    total=arms_counter.sum()
                    var=torch.mean(torch.pow(q-mean.unsqueeze(2),2),dim=2)
                    upper=torch.sqrt(2*(var)*torch.log(total)/arms_counter)
                    upperBound=mean+upper
                    return torch.argmax(upperBound,dim=1)
            def tderror(self,transition,pris):
                #print(transition)
                state,action,next_state,reward,done=transition
                
                new_value=self.forward(next_state)
                new_q=new_value.mean(dim=2)
                new_act=torch.argmax(new_q,dim=1)
                # print(new_act.shape) #[batch]
                # print(self.target_foward(next_state).shape) [batch,act_space,quantiles]
                # print(new_q.shape)
                # print(self.target_foward(next_state)[:,new_act,:].shape)
                # print((1-done.float()).shape)
                tfrd=self.target_foward(next_state)
                next_z=torch.stack([tfrd[i,new_act[i],:] for i in range(len(new_act))],dim=0)
                #print(next_z.shape,(1-done.float()).shape,reward.shape)
                target_z=(self.gamma*next_z*(1-done.float())+reward.unsqueeze(1)).detach()
                frd=self.forward(state)
                
                z_now=torch.stack([frd[i,action[i],:] for i in range(len(action))],dim=0)
                #print(z_now.shape,target_z.shape,self.qvs.shape)
                z_now=z_now[:,0,:]
                _tderror=self.loss(self.qvs,target_z,z_now)
                new_priorities=torch.abs(_tderror)+1e-6
                
                return torch.mean(pris[:,0]*_tderror),new_priorities.detach()
        
        self.transition=namedtuple('transition',['state','action','next_state','reward','done'])
        self.epsilon=epsilon
        self.tau=tau
        self.buffer_size=buffer_size
        self.update_inter=update_inter
        self.lr=lr
        self.arms=arms
        self.agents_num=agents_num
        self.batch_size=batch_size
        self.quantiles=quantiles
        self.net=NetModule(context_len=5,act_space=arms,quantiles=quantiles,epsilon=epsilon)
        self.replay_buffer=replayBuffer(buffer_size,batch_size)
        #self.last_state,self.last_action,self.last_reward,self.last_done=None,None,None,None
        self.train_counter=0
        self.soft_counter=0
        self.WINDOWS=5000
        self.opt=torch.optim.Adam(self.net.parameters(),lr=lr)
        self.arms_counter=torch.zeros((self.WINDOWS,self.arms))
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        
    def act(self,Q,Q_dist,quos,cost,info=None):
        Q,Q_dist = None,None
        assert info != None
        context=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(1,-1)
        state=torch.from_numpy(context).float()
        act=self.net.act(state,arms_counter=self.arms_counter)
        # if self.last_state is not None:
        #     tran=self.transition(self.last_state,self.last_action,state,self.last_reward,False)
        # self.last_state,self.last_action=context,act
        temp_vec=torch.zeros((1,self.arms))
        temp_vec[0,act]=1
        self.arms_counter=torch.concatenate([self.arms_counter[1:,:],temp_vec],axis=0)
        act=self.arms_quantile[act]
        return act
        
    def updating(self,transition):
        state,action,next_state,reward,done=transition
        reward=reward[0]
        transition=self.transition(state,action,next_state,reward,done)
        #print(transition)
        self.replay_buffer.add(transition)
        self.train_counter+=1
        if self.train_counter % (self.batch_size)==0:
            #print("training",self.train_counter)
            batch,idxs,pris=self.replay_buffer.sample()
            state,action,next_state,reward,done=batch
            action=((action+1)/(self.arms_quantile[1]-self.arms_quantile[0])).int()
            batch=[state,action,next_state,reward,done]
            #print("sampled",batch)
            loss,new_prior=self.net.tderror(batch,pris)
            #print("loss",loss,"prior",new_prior)
            for idx,new_pri in zip(idxs,new_prior):
                self.replay_buffer.update(idx,new_pri)
            self.net.zero_grad()
            loss.backward()
            self.opt.step()
            self.soft_counter+=1
            if self.soft_counter%self.tau==0:
                self.net.update_target()

@ray.remote(num_cpus=1)
class DistributionalD3QNVUCBAgentWithPER():
    def __init__(self,arms,agents_num=[0],update_inter=10,quantiles=10,lr=0.005,epsilon=0.05,tau=10,buffer_size=1000,batch_size=16):
        # distributional double dueling deep Q network
        class replayBuffer():
            def __init__(self,capacity,batch_size,alpha=0.1,beta=0.1,beta_increment_per_sampling=0.001):
                self.e=0.001
                self.capacity=capacity
                self.batch_size=batch_size
                self.alpha=alpha
                self.beta=beta
                self.beta_increment_per_sampling=beta_increment_per_sampling
                
                class Sum_Tree():
                    
                    def __init__(self,capacity):
                        self.tree=np.zeros(2*capacity-1)
                        self.data=np.zeros(capacity,dtype=object)
                        self.entries=0
                        self.write=0
                        self.capacity=capacity
                        self.max_prio=10
                    def _propagate(self, idx, change):
                        parent = (idx - 1) // 2
                        self.tree[parent] += change
                        if parent != 0:
                            self._propagate(parent, change)
                            
                    def _retrieve(self, idx, s):
                        left = 2 * idx + 1
                        right = left + 1
                        if left >= len(self.tree):
                            return idx
                        if s <= self.tree[left]:
                            return self._retrieve(left, s)
                        else:
                            return self._retrieve(right, s - self.tree[left])
                        
                    def _update(self,idx,p):
                        change=p-self.tree[idx]
                        self.tree[idx]=p
                        self._propagate(idx,change)
                        if p>self.max_prio:
                            self.max_prio=p
                    def _add(self, p,data):
                        idx=self.write+self.capacity-1
                        self.data[self.write]=data
                        self._update(idx,p)
                        self.write+=1
                        if self.write>=self.capacity:
                            self.write=0
                        if self.entries<self.capacity:
                            self.entries+=1
                        if p>self.max_prio:
                            self.max_prio=p
                            
                    def _get(self,s):
                        idx=self._retrieve(0,s)
                        dataidx=idx-self.capacity+1
                        return (idx,self.tree[idx],self.data[dataidx])
                    
                    def total(self):
                        return self.tree[0]
                    
                self.tree=Sum_Tree(capacity)
                
            def _get_priority(self,error):
                return (np.abs(error)+self.e)**self.alpha
            
            def add(self,tran,error=None):
                if error is None:
                    p=self.tree.max_prio
                else:
                    p=self._get_priority(error)

                self.tree._add(p,tran)

            def sample(self,batchs=0):
                if batchs==0:
                    batchs=self.batch_size
                sampled=[]
                idxs=[]
                pris=[]
                segment=self.tree.total()/batchs
                
                self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
                for _ in range(batchs):
                    a=segment*_
                    b=segment*(_+1)
                    s=np.random.uniform(a,b)
                    (idx,p,data)=self.tree._get(s)
                    idxs.append(idx)
                    pris.append(torch.tensor([p],dtype=float))
                    sampled.append(data)
                #print(sampled)
                transed=list(zip(*sampled))
                for i in range(len(transed)):
                    transed[i]=torch.stack(transed[i],dim=0)
                pris=torch.stack(pris,dim=0).float()
                pris=pris/self.tree.total()
                weights=torch.pow(self.tree.entries*pris,-self.beta)
                weights=weights/torch.max(weights,dim=0,keepdim=True)[0]
                #print(transed,idxs,weights)
                return transed,idxs,weights
            
            def update(self,idx,error):
                p=self._get_priority(error)
                self.tree._update(idx,p)

        class NetModule(nn.Module):
            def __init__(self,context_len,act_space,quantiles,dim1=32,dim2=64,dim3=32,gamma=0.99,epsilon=0.05):
                super(NetModule,self).__init__()
                self.quantiles=quantiles
                self.act_space=act_space
                self.context_len=context_len
                self.gamma=gamma
                self.hidden1=nn.Sequential(
                    nn.Linear(context_len,dim1),
                    nn.ReLU(),
                    nn.Linear(dim1,dim2),
                    nn.ReLU()
                )
                self.A1=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,act_space*quantiles)
                )
                self.V1=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,quantiles)
                )
                
                self.hidden2=nn.Sequential(
                    nn.Linear(context_len,dim1),
                    nn.ReLU(),
                    nn.Linear(dim1,dim2),
                    nn.ReLU()
                )
                self.A2=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,act_space*quantiles)
                )
                self.V2=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,quantiles)
                )
                self.qvs=torch.linspace(0.0,1.0-1/(quantiles),quantiles)+1/(2*quantiles)
                
                def loss(quan,target,pred,delta=0.5):
                    u=target-pred
                    weights=torch.abs(quan-(u<0).float()).detach()
                    huber=nn.functional.huber_loss(pred,target,reduction='none',delta=delta)
                    return (weights*huber).mean(dim=1)
                
                self.loss=loss
                self.epsilon=epsilon
            def forward(self,context):
                context=self.hidden1(context)
                v=self.V1(context) # batch,quantiles
                a=self.A1(context) # batch,act_space*quantiles
                a=a.view(-1,self.act_space,self.quantiles) # batch,act_space,quantiles
                a=torch.abs(a)
                a=a.cumsum(dim=2)
                v=v.view(-1,1,self.quantiles)# batch,1,quantiles
                v=torch.abs(v)
                v=v.cumsum(dim=2)
                q=torch.add(v,a)-torch.mean(a,dim=1,keepdim=True) # batch,act_space,quantiles
                return q
            
            def target_foward(self,context):
                context=self.hidden2(context)
                v=self.V2(context) # batch,quantiles
                a=self.A2(context) # batch,act_space*quantiles
                a=a.view(-1,self.act_space,self.quantiles) # batch,act_space,quantiles
                v=v.view(-1,1,self.quantiles) # batch,1,quantiles
                q=torch.add(v,a)-torch.mean(a,dim=1,keepdim=True) # batch,act_space,quantiles
                return q
            
            def update_target(self,tau=0.2):
                for target,param in zip(self.hidden2.parameters(),self.hidden1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                for target,param in zip(self.A2.parameters(),self.A1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                for target,param in zip(self.V2.parameters(),self.V1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                
            def act(self,context,stra='VarianceUCB',arms_counter=None,windows=1000):
                q=self.forward(context)
                if stra=='epsilon_greedy':
                    if np.random.rand()<0.1:
                        return torch.randint(0,self.act_space,(1,))
                    else:
                        return torch.argmax(torch.mean(q,dim=2),dim=1)
                if stra=='simpleUCB':
                    assert arms_counter is not None
                    CONSTANT=1
                    arms_counter=arms_counter.sum(dim=0)
                    upper=CONSTANT*torch.sqrt(2*torch.log(total)/arms_counter)
                    mean=torch.mean(q,dim=2)
                    upperBound=mean+upper
                    return torch.argmax(upperBound,dim=1)
                if stra=='VarianceUCB':
                    #calculate v sub gaussian
                    mean=torch.mean(q,dim=2)
                    arms_counter=torch.sum(arms_counter,dim=0)
                    total=arms_counter.sum()
                    var=torch.mean(torch.pow(q-mean.unsqueeze(2),2),dim=2)
                    upper=torch.sqrt(2*(var)*torch.log(total)/arms_counter)
                    upperBound=mean+upper
                    return torch.argmax(upperBound,dim=1)
            def tderror(self,transition,pris):
                #print(transition)
                state,action,next_state,reward,done=transition
                
                new_value=self.forward(next_state)
                new_q=new_value.mean(dim=2)
                new_act=torch.argmax(new_q,dim=1)
                # print(new_act.shape) #[batch]
                # print(self.target_foward(next_state).shape) [batch,act_space,quantiles]
                # print(new_q.shape)
                # print(self.target_foward(next_state)[:,new_act,:].shape)
                # print((1-done.float()).shape)
                tfrd=self.target_foward(next_state)
                next_z=torch.stack([tfrd[i,new_act[i],:] for i in range(len(new_act))],dim=0)
                #print(next_z.shape,(1-done.float()).shape,reward.shape)
                target_z=(self.gamma*next_z*(1-done.float())+reward.unsqueeze(1)).detach()
                frd=self.forward(state)
                
                z_now=torch.stack([frd[i,action[i],:] for i in range(len(action))],dim=0)
                #print(z_now.shape,target_z.shape,self.qvs.shape)
                z_now=z_now[:,0,:]
                _tderror=self.loss(self.qvs,target_z,z_now)
                new_priorities=torch.abs(_tderror)+1e-6
                
                return torch.mean(pris[:,0]*_tderror),new_priorities.detach()
        
        self.transition=namedtuple('transition',['state','action','next_state','reward','done'])
        self.epsilon=epsilon
        self.tau=tau
        self.buffer_size=buffer_size
        self.update_inter=update_inter
        self.lr=lr
        self.arms=arms
        self.agents_num=agents_num
        self.batch_size=batch_size
        self.quantiles=quantiles
        self.net=NetModule(context_len=5,act_space=arms,quantiles=quantiles,epsilon=epsilon)
        self.replay_buffer=replayBuffer(buffer_size,batch_size)
        #self.last_state,self.last_action,self.last_reward,self.last_done=None,None,None,None
        self.WINDOWS=5000 #! windows
        self.train_counter=0
        self.soft_counter=0
        self.opt=torch.optim.Adam(self.net.parameters(),lr=lr)
        self.arms_counter=torch.zeros((self.WINDOWS,self.arms))
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        
        
    def act(self,Q,Q_dist,quos,cost,info=None):
        Q,Q_dist = None,None
        assert info != None
        context=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(1,-1)
        state=torch.from_numpy(context).float()
        act=self.net.act(state,arms_counter=self.arms_counter,windows=self.WINDOWS)
        # if self.last_state is not None:
        #     tran=self.transition(self.last_state,self.last_action,state,self.last_reward,False)
        # self.last_state,self.last_action=context,act
        temp_vec=torch.zeros((1,self.arms))
        temp_vec[0,act]=1
        self.arms_counter=torch.concatenate([self.arms_counter[1:,:],temp_vec],axis=0)
        act=self.arms_quantile[act]
        return act
        
    def updating(self,transition):
        state,action,next_state,reward,done=transition
        reward=reward[0]
        transition=self.transition(state,action,next_state,reward,done)
        #print(transition)
        self.replay_buffer.add(transition)
        self.train_counter+=1
        if self.train_counter%(self.batch_size)==0:
            batch,idxs,pris=self.replay_buffer.sample()
            state,action,next_state,reward,done=batch
            action=((action+1)/(self.arms_quantile[1]-self.arms_quantile[0])).int()
            batch=[state,action,next_state,reward,done]
            #print("sampled",batch)
            loss,new_prior=self.net.tderror(batch,pris)
            #print("loss",loss,"prior",new_prior)
            for idx,new_pri in zip(idxs,new_prior):
                self.replay_buffer.update(idx,new_pri)
            self.net.zero_grad()
            loss.backward()
            self.opt.step()
            self.soft_counter+=1
            if self.soft_counter%self.tau==0:
                self.net.update_target()
                
@ray.remote(num_cpus=1)
class DistributionalD3QNEAgentWithPER():
    def __init__(self,arms,agents_num=[0],update_inter=10,quantiles=10,lr=0.005,epsilon=0.2,tau=10,buffer_size=1000,batch_size=16):
        # distributional double dueling deep Q network
        class replayBuffer():
            def __init__(self,capacity,batch_size,alpha=0.1,beta=0.1,beta_increment_per_sampling=0.001):
                self.e=0.001
                self.capacity=capacity
                self.batch_size=batch_size
                self.alpha=alpha
                self.beta=beta
                self.beta_increment_per_sampling=beta_increment_per_sampling
                
                class Sum_Tree():
                    
                    def __init__(self,capacity):
                        self.tree=np.zeros(2*capacity-1)
                        self.data=np.zeros(capacity,dtype=object)
                        self.entries=0
                        self.write=0
                        self.capacity=capacity
                        self.max_prio=10
                    def _propagate(self, idx, change):
                        parent = (idx - 1) // 2
                        self.tree[parent] += change
                        if parent != 0:
                            self._propagate(parent, change)
                            
                    def _retrieve(self, idx, s):
                        left = 2 * idx + 1
                        right = left + 1
                        if left >= len(self.tree):
                            return idx
                        if s <= self.tree[left]:
                            return self._retrieve(left, s)
                        else:
                            return self._retrieve(right, s - self.tree[left])
                        
                    def _update(self,idx,p):
                        change=p-self.tree[idx]
                        self.tree[idx]=p
                        self._propagate(idx,change)
                        if p>self.max_prio:
                            self.max_prio=p
                    def _add(self, p,data):
                        idx=self.write+self.capacity-1
                        self.data[self.write]=data
                        self._update(idx,p)
                        self.write+=1
                        if self.write>=self.capacity:
                            self.write=0
                        if self.entries<self.capacity:
                            self.entries+=1
                        if p>self.max_prio:
                            self.max_prio=p
                            
                    def _get(self,s):
                        idx=self._retrieve(0,s)
                        dataidx=idx-self.capacity+1
                        return (idx,self.tree[idx],self.data[dataidx])
                    
                    def total(self):
                        return self.tree[0]
                    
                self.tree=Sum_Tree(capacity)
                
            def _get_priority(self,error):
                return (np.abs(error)+self.e)**self.alpha
            
            def add(self,tran,error=None):
                if error is None:
                    p=self.tree.max_prio
                else:
                    p=self._get_priority(error)

                self.tree._add(p,tran)

            def sample(self,batchs=0):
                if batchs==0:
                    batchs=self.batch_size
                sampled=[]
                idxs=[]
                pris=[]
                segment=self.tree.total()/batchs
                
                self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
                for _ in range(batchs):
                    a=segment*_
                    b=segment*(_+1)
                    s=np.random.uniform(a,b)
                    (idx,p,data)=self.tree._get(s)
                    idxs.append(idx)
                    pris.append(torch.tensor([p],dtype=float))
                    sampled.append(data)
                #print(sampled)
                transed=list(zip(*sampled))
                for i in range(len(transed)):
                    transed[i]=torch.stack(transed[i],dim=0)
                pris=torch.stack(pris,dim=0).float()
                pris=pris/self.tree.total()
                weights=torch.pow(self.tree.entries*pris,-self.beta)
                weights=weights/torch.max(weights,dim=0,keepdim=True)[0]
                #print(transed,idxs,weights)
                return transed,idxs,weights
            
            def update(self,idx,error):
                p=self._get_priority(error)
                self.tree._update(idx,p)

        class NetModule(nn.Module):
            def __init__(self,context_len,act_space,quantiles,dim1=32,dim2=64,dim3=32,gamma=0.99,epsilon=0.05):
                super(NetModule,self).__init__()
                self.quantiles=quantiles
                self.act_space=act_space
                self.context_len=context_len
                self.gamma=gamma
                self.hidden1=nn.Sequential(
                    nn.Linear(context_len,dim1),
                    nn.ReLU(),
                    nn.Linear(dim1,dim2),
                    nn.ReLU()
                )
                self.A1=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,act_space*quantiles)
                )
                self.V1=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,quantiles)
                )
                
                self.hidden2=nn.Sequential(
                    nn.Linear(context_len,dim1),
                    nn.ReLU(),
                    nn.Linear(dim1,dim2),
                    nn.ReLU()
                )
                self.A2=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,act_space*quantiles)
                )
                self.V2=nn.Sequential(
                    nn.Linear(dim2,dim3),
                    nn.ReLU(),
                    nn.Linear(dim3,quantiles)
                )
                self.qvs=torch.linspace(0.0,1.0-1/(quantiles),quantiles)+1/(2*quantiles)
                self.epsilon=epsilon
                
                def loss(quan,target,pred,delta=0.5):
                    u=target-pred
                    weights=torch.abs(quan-(u<0).float()).detach()
                    huber=nn.functional.huber_loss(pred,target,reduction='none',delta=delta)
                    return (weights*huber).mean(dim=1)
                
                self.loss=loss
                
            def forward(self,context):
                context=self.hidden1(context)
                v=self.V1(context) # batch,quantiles
                a=self.A1(context) # batch,act_space*quantiles
                a=a.view(-1,self.act_space,self.quantiles) # batch,act_space,quantiles
                a=torch.abs(a)
                a=a.cumsum(dim=2)
                v=v.view(-1,1,self.quantiles)# batch,1,quantiles
                v=torch.abs(v)
                v=v.cumsum(dim=2)
                q=torch.add(v,a)-torch.mean(a,dim=1,keepdim=True) # batch,act_space,quantiles
                return q
            
            def target_foward(self,context):
                context=self.hidden2(context)
                v=self.V2(context) # batch,quantiles
                a=self.A2(context) # batch,act_space*quantiles
                a=a.view(-1,self.act_space,self.quantiles) # batch,act_space,quantiles
                v=v.view(-1,1,self.quantiles) # batch,1,quantiles
                q=torch.add(v,a)-torch.mean(a,dim=1,keepdim=True) # batch,act_space,quantiles
                return q
            
            def update_target(self,tau=0.2):
                for target,param in zip(self.hidden2.parameters(),self.hidden1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                for target,param in zip(self.A2.parameters(),self.A1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                for target,param in zip(self.V2.parameters(),self.V1.parameters()):
                    target.data=tau*target.data+(1-tau)*param.data
                
            def act(self,context,stra='epsilon_greedy',arms_counter=None):
                q=self.forward(context)
                if stra=='epsilon_greedy':
                    if np.random.rand()<self.epsilon*torch.exp(-arms_counter.sum()/10000):
                        return torch.randint(0,self.act_space,(1,))
                    else:
                        return torch.argmax(torch.mean(q,dim=2),dim=1)
                if stra=='simpleUCB':
                    assert arms_counter is not None
                    CONSTANT=1
                    total=arms_counter.sum()
                    upper=CONSTANT*torch.sqrt(2*torch.log(total)/arms_counter)
                    mean=torch.mean(q,dim=2)
                    upperBound=mean+upper
                    return torch.argmax(upperBound,dim=1)
                if stra=='VarianceUCB':
                    #calculate v sub gaussian
                    mean=torch.mean(q,dim=2)
                    total=arms_counter.sum()
                    var=torch.mean(torch.pow(q-mean.unsqueeze(2),2),dim=2)
                    upper=torch.sqrt(2*(var)*torch.log(total)/arms_counter)
                    upperBound=mean+upper
                    return torch.argmax(upperBound,dim=1)
            def tderror(self,transition,pris):
                #print(transition)
                state,action,next_state,reward,done=transition
                
                new_value=self.forward(next_state)
                new_q=new_value.mean(dim=2)
                new_act=torch.argmax(new_q,dim=1)
                # print(new_act.shape) #[batch]
                # print(self.target_foward(next_state).shape) [batch,act_space,quantiles]
                # print(new_q.shape)
                # print(self.target_foward(next_state)[:,new_act,:].shape)
                # print((1-done.float()).shape)
                tfrd=self.target_foward(next_state)
                next_z=torch.stack([tfrd[i,new_act[i],:] for i in range(len(new_act))],dim=0)
                #print(next_z.shape,(1-done.float()).shape,reward.shape)
                target_z=(self.gamma*next_z*(1-done.float())+reward.unsqueeze(1)).detach()
                frd=self.forward(state)
                
                z_now=torch.stack([frd[i,action[i],:] for i in range(len(action))],dim=0)
                #print(z_now.shape,target_z.shape,self.qvs.shape)
                z_now=z_now[:,0,:]
                _tderror=self.loss(self.qvs,target_z,z_now)
                new_priorities=torch.abs(_tderror)+1e-6
                
                return torch.mean(pris[:,0]*_tderror),new_priorities.detach()
        
        self.transition=namedtuple('transition',['state','action','next_state','reward','done'])
        self.epsilon=epsilon
        self.tau=tau
        self.buffer_size=buffer_size
        self.update_inter=update_inter
        self.lr=lr
        self.arms=arms
        self.agents_num=agents_num
        self.batch_size=batch_size
        self.quantiles=quantiles
        self.net=NetModule(context_len=5,act_space=arms,quantiles=quantiles,epsilon=epsilon)
        self.replay_buffer=replayBuffer(buffer_size,batch_size)
        #self.last_state,self.last_action,self.last_reward,self.last_done=None,None,None,None
        self.train_counter=0
        self.soft_counter=0
        self.opt=torch.optim.Adam(self.net.parameters(),lr=lr)
        self.arms_counter=torch.zeros(arms)
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        
    def act(self,Q,Q_dist,quos,cost,info=None):
        Q,Q_dist = None,None
        assert info != None
        context=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(1,-1)
        state=torch.from_numpy(context).float()
        act=self.net.act(state,arms_counter=self.arms_counter)
        # if self.last_state is not None:
        #     tran=self.transition(self.last_state,self.last_action,state,self.last_reward,False)
        # self.last_state,self.last_action=context,act
        self.arms_counter[act]+=1
        act=self.arms_quantile[act]
        return act
        
    def updating(self,transition):
        state,action,next_state,reward,done=transition
        reward=reward[0]
        transition=self.transition(state,action,next_state,reward,done)
        #print(transition)
        self.replay_buffer.add(transition)
        self.train_counter+=1
        if self.train_counter%(self.batch_size)==0:
            batch,idxs,pris=self.replay_buffer.sample()
            state,action,next_state,reward,done=batch
            action=((action+1)/(self.arms_quantile[1]-self.arms_quantile[0])).int()
            batch=[state,action,next_state,reward,done]
            #print("sampled",batch)
            loss,new_prior=self.net.tderror(batch,pris)
            #print("loss",loss,"prior",new_prior)
            for idx,new_pri in zip(idxs,new_prior):
                self.replay_buffer.update(idx,new_pri)
            self.net.zero_grad()
            loss.backward()
            self.opt.step()
            self.soft_counter+=1
            if self.soft_counter%self.tau==0:
                self.net.update_target()

#@ray.remote(num_cpus=1)
class End2EndSACAgent():
    def __init__(self,quantiles=16,max_act=200,hidden_dim=128,agents_num=[0],update_inter=10,lr=0.005,epsilon=0.2,tau=0.1,buffer_size=1000,batch_size=16,gamma=0.99):
        # SAC
        class replayBuffer():
            def __init__(self,capacity,batch_size,alpha=0.1,beta=0.1,beta_increment_per_sampling=0.001):
                self.e=0.001
                self.capacity=capacity
                self.batch_size=batch_size
                self.alpha=alpha
                self.beta=beta
                self.beta_increment_per_sampling=beta_increment_per_sampling
                
                class Sum_Tree():
                    
                    def __init__(self,capacity):
                        self.tree=np.zeros(2*capacity-1)
                        self.data=np.zeros(capacity,dtype=object)
                        self.entries=0
                        self.write=0
                        self.capacity=capacity
                        self.max_prio=10
                    def _propagate(self, idx, change):
                        parent = (idx - 1) // 2
                        self.tree[parent] += change
                        if parent != 0:
                            self._propagate(parent, change)
                            
                    def _retrieve(self, idx, s):
                        left = 2 * idx + 1
                        right = left + 1
                        if left >= len(self.tree):
                            return idx
                        if s <= self.tree[left]:
                            return self._retrieve(left, s)
                        else:
                            return self._retrieve(right, s - self.tree[left])
                        
                    def _update(self,idx,p):
                        change=p-self.tree[idx]
                        self.tree[idx]=p
                        self._propagate(idx,change)
                        if p>self.max_prio:
                            self.max_prio=p
                    def _add(self, p,data):
                        idx=self.write+self.capacity-1
                        self.data[self.write]=data
                        self._update(idx,p)
                        self.write+=1
                        if self.write>=self.capacity:
                            self.write=0
                        if self.entries<self.capacity:
                            self.entries+=1
                        if p>self.max_prio:
                            self.max_prio=p
                            
                    def _get(self,s):
                        idx=self._retrieve(0,s)
                        dataidx=idx-self.capacity+1
                        return (idx,self.tree[idx],self.data[dataidx])
                    
                    def total(self):
                        return self.tree[0]
                    
                self.tree=Sum_Tree(capacity)
                
            def _get_priority(self,error):
                return (np.abs(error)+self.e)**self.alpha
            
            def add(self,tran,error=None):
                if error is None:
                    p=self.tree.max_prio
                else:
                    p=self._get_priority(error)

                self.tree._add(p,tran)

            def sample(self,batchs=0):
                if batchs==0:
                    batchs=self.batch_size
                sampled=[]
                idxs=[]
                pris=[]
                segment=self.tree.total()/batchs
                
                self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
                for _ in range(batchs):
                    a=segment*_
                    b=segment*(_+1)
                    s=np.random.uniform(a,b)
                    (idx,p,data)=self.tree._get(s)
                    idxs.append(idx)
                    pris.append(torch.tensor([p],dtype=float))
                    sampled.append(data)
                #print(sampled)
                transed=list(zip(*sampled))
                for i in range(len(transed)):
                    transed[i]=torch.stack(transed[i],dim=0)
                pris=torch.stack(pris,dim=0).float()
                pris=pris/self.tree.total()
                weights=torch.pow(self.tree.entries*pris,-self.beta)
                weights=weights/torch.max(weights,dim=0,keepdim=True)[0]
                #print(transed,idxs,weights)
                return transed,idxs,weights
            
            def update(self,idx,error):
                p=self._get_priority(error)
                self.tree._update(idx,p)
        
        class Shared_layers(nn.Module):
            def __init__(self,context_len,out_len=128):
                super(Shared_layers,self).__init__()
                self.hidden1=nn.Sequential(
                    nn.Linear(context_len,32),
                    nn.ReLU(),
                    nn.Linear(32,out_len),
                    nn.ReLU()
                )
            def forward(self,context):
                return self.hidden1(context)
                
        class Actor(nn.Module):
            def __init__(self,context_len,act_bound=200):
                super(Actor,self).__init__()
                self.hidden1=nn.Sequential(
                    nn.Linear(context_len,32),
                    nn.ReLU(),
                    nn.Linear(32,64),
                    nn.ReLU(),
                )
                self.log_std_l=nn.Linear(64,1)
                self.mean_l=nn.Linear(64,1)
                self.act_bound=act_bound
            def forward(self,context):
                x=self.hidden1(context)
                mean=self.mean_l(x)
                log_std=torch.clamp(self.log_std_l(x),-20,2)
                std=torch.exp(log_std)
                dist=torch.distributions.Normal(mean,std)
                #reparametrization trick
                act=dist.rsample()
                log_prob= dist.log_prob(act).sum(dim=-1,keepdim=True)
                log_prob-=(2*(np.log(2)-act-torch.nn.functional.softplus(-2*act))).sum(dim=-1,keepdim=True)
                act=self.act_bound*torch.sigmoid(act) + 1 #! 1 is the minimum action
                return act,log_prob
        
        class Critic(nn.Module):
            def __init__(self, context_len, act_dim=1,quantiles=16):
                super(Critic, self).__init__()
                self.quantiles=torch.tensor([i/quantiles for i in range(1,quantiles+1)],dtype=torch.float32)+1/(2*quantiles)
                self.Q1=nn.Sequential(
                    nn.Linear(context_len+act_dim,32),
                    nn.ReLU(),
                    nn.Linear(32,64),
                    nn.ReLU(),
                    nn.Linear(64,quantiles)
                )
                self.Q2=nn.Sequential(
                    nn.Linear(context_len+act_dim,32),
                    nn.ReLU(),
                    nn.Linear(32,64),
                    nn.ReLU(),
                    nn.Linear(64,quantiles)
                )
            def forward(self,context,a):
                inp=torch.cat([context,a],dim=1)
                return self.Q1(inp),self.Q2(inp)
            

        self.transition=namedtuple('transition',['state','action','next_state','reward','done'])
        self.epsilon=epsilon
        self.tau=tau
        self.buffer_size=buffer_size
        self.update_inter=update_inter
        self.lr=lr
        self.max_act=max_act
        self.agents_num=agents_num
        self.batch_size=batch_size
        self.action_dim=1
        self.gamma=gamma
        self.replay_buffer=replayBuffer(buffer_size,batch_size)
        #self.last_state,self.last_action,self.last_reward,self.last_done=None,None,None,None
        self.train_counter=0
        self.soft_counter=0
        self.context_len=10
        self.hidden_dim=hidden_dim
        self.target_entropy=-self.action_dim
        self.quantiles=torch.tensor([i/quantiles for i in range(1,quantiles+1)],dtype=torch.float32)+1/(2*quantiles)
        
        self.shared=Shared_layers(self.context_len,self.hidden_dim)
        self.actor=Actor(self.hidden_dim,self.max_act)
        self.critic=Critic(self.hidden_dim,self.action_dim,quantiles=quantiles)
        self.target_critic=copy.deepcopy(self.critic)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        
        self.opt_shared=optim.Adam(self.shared.parameters(),lr=self.lr)
        self.opt_actor=optim.Adam(self.actor.parameters(),lr=self.lr)
        self.opt_critic=optim.Adam(self.critic.parameters(),lr=self.lr)
        self.opt_alpha=optim.Adam([self.log_alpha],lr=self.lr)
        
        def loss(target,pred,delta=0.5,quan=self.quantiles): 
            #print("target",target.shape,"pred",pred.shape,"quan",quan.shape)
            target=target.unsqueeze(2)
            pred=pred.unsqueeze(1)
            u=target-pred
            #print("u",u.shape,u)
            weights=torch.abs(quan-(u<0).float()).detach()
            #print("weights",weights.shape,weights)
            huber=nn.functional.huber_loss(u,torch.zeros_like(u),reduction='none',delta=delta)
            lor=(weights*huber).mean(dim=1).sum(dim=1)
            #print("lor",lor)
            return lor
        self.tdlossf=loss
        
    def soft_update(self,net,target_net):
        for target_param,param in zip(target_net.parameters(),net.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

    
    def get_loss(self,trans,pris):
        state,action,next_state,reward,done=trans
        #print('tran',trans)

        LOSS=0
        with torch.no_grad():
            next_state=self.shared(next_state) 
            next_action,next_log_prob=self.actor(next_state)
            target_Q1,target_Q2=self.target_critic(next_state,next_action)
            target_Q=torch.min(target_Q1,target_Q2)
            tdtar=reward+self.gamma*(1-done.float())*(target_Q-next_log_prob*self.log_alpha.exp().detach())
            
        state=self.shared(state)
        current_Q1,current_Q2=self.critic(state,action)
        tdloss=0.5*(self.tdlossf(current_Q1,tdtar)+self.tdlossf(current_Q2,tdtar))
        new_pris=torch.abs(tdloss).detach()+1e-6
        LOSS+=torch.mean(tdloss*pris)
        
        act,log_prob=self.actor(state)
        Qnow=torch.min(current_Q1,current_Q2).mean(dim=1,keepdim=True)
        
        act_loss=(self.log_alpha.exp().detach()*log_prob-Qnow)
        LOSS+=torch.mean(act_loss*pris)
        
        alpha_loss=-self.log_alpha*(log_prob+self.target_entropy).detach()
        LOSS+=torch.mean(alpha_loss*pris)
        
        return LOSS,new_pris
    
    @torch.no_grad()
    def act(self,Q,Q_dist,quos,cost,info=None):
        #Q,Q_dist = None,None
        Q_dist = None
        assert info != None
        context=np.concatenate([Q.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(1,-1)
        state=torch.from_numpy(context).float()
        
        context=self.shared(state)
        act,log_prob=self.actor(context)
        
        return act
        
    def updating(self,transition):
        state,action,next_state,reward,done=transition
        reward=reward[0].unsqueeze(0)
        action=action[0]
        transition=self.transition(state,action,next_state,reward,done)
        #print(transition)
        self.replay_buffer.add(transition)
        self.train_counter+=1
        if self.train_counter%(self.batch_size)==0:
            batch,idxs,pris=self.replay_buffer.sample()
            state,action,next_state,reward,done=batch
            #print("sampled",batch)
            loss,new_prior=self.get_loss(batch,pris)
            #print("loss",loss,"prior",new_prior)
            for idx,new_pri in zip(idxs,new_prior):
                self.replay_buffer.update(idx,new_pri)
            
            self.opt_shared.zero_grad()
            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()
            self.opt_alpha.zero_grad()
            loss.backward()
            self.opt_shared.step()
            self.opt_actor.step()
            self.opt_critic.step()
            self.opt_alpha.step()
            
            self.soft_counter+=1
            if self.soft_counter%self.tau==0:
                self.soft_update(self.critic,self.target_critic)
                
class E2EmultiAgents():
    def __init__(self,agents,arg_str):
        self.agents=agents
        self.arg_str=arg_str
    def acts(self,Q,Q_dist,quos,cost,info=None):
        handle=[]
        for i,agent in enumerate(self.agents):
            dri_nums=info['driver_nums'][i]
            handle.append(agent.act.remote(Q[i],Q_dist[i],quos[i],cost[i]*configure.cost_rate[0],dri_nums[0:len(configure.cost_rate)]))
        actions=ray.get(handle)
        return torch.stack(actions)
    
    def updating(self,transitions):
        for i,agent in enumerate(self.agents):
            agent.updating.remote(transitions[i])
            
    def log(self,writter):
        for i,agent in enumerate(self.agents):
            agent.log.remote(self.arg_str[i],writter)
            
    def obs2Context(self,Q,Q_dist,quos,cost,info=None):
        ctxs=[]
        for i in range(len(self.agents)):
            dri_nums=info['driver_nums'][i]
            context=np.array([*Q[i].numpy(),*cost[i].unsqueeze(0).numpy()*configure.cost_rate[0],dri_nums[i]]).reshape(1,-1)
            ctxs.append(torch.from_numpy(context).float())
        ctx=torch.cat(ctxs,dim=0)
        return ctx

class Env2multiAgents():
    def __init__(self,agents,arg_str):
        self.agents=agents
        self.arg_str=arg_str
    def acts(self,Q,Q_dist,quos,cost,info=None):
        handle=[]
        for i,agent in enumerate(self.agents):
            dri_nums=info['driver_nums'][i]
            handle.append(agent.act.remote(Q[i],Q_dist[i],quos[i],cost[i]*configure.cost_rate[0],dri_nums[0:len(configure.cost_rate)]))
        actions=ray.get(handle)
        return torch.stack(actions)
    
    def updating(self,transitions):
        for i,agent in enumerate(self.agents):
            agent.updating.remote(transitions[i])
            
    def log(self,writter):
        for i,agent in enumerate(self.agents):
            agent.log.remote(self.arg_str[i],writter)
            
    def obs2Context(self,Q,Q_dist,quos,cost,info=None):
        ctxs=[]
        for i in range(len(self.agents)):
            dri_nums=info['driver_nums'][i]
            context=np.array([*quos[i].numpy(),*cost[i].unsqueeze(0).numpy()*configure.cost_rate[0],dri_nums[i]]).reshape(1,-1)
            ctxs.append(torch.from_numpy(context).float())
        ctx=torch.cat(ctxs,dim=0)
        return ctx



@ray.remote(num_cpus=1)
class multiQuoMultiArmBanditAgentIn2():
    def __init__(self,arms,agents_num=len(configure.cost_rate)):
        self.arms=arms
        self.agents_num=agents_num
        self.counts=[torch.zeros(arms,dtype=torch.float32) for _ in range(agents_num)]
        self.values=[torch.zeros(arms,dtype=torch.float32) for _ in range(agents_num)]
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        self.cost_rate=configure.cost_rate
        
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
    
    def act(self,Q,Q_dist,quos,cost,info=None):
        temp=[]
        for q in range(len(self.arms_quantile)):
            temp.append(self.act_to_quo(quos,self.arms_quantile[q]))
        act_quos=torch.tensor(temp,dtype=torch.float32)
        act=[]
        for i in range(self.agents_num):
            if torch.any(self.counts[i])==0:
                act.append(self.arms_quantile[torch.argwhere(self.counts[i]==0)[0]])
            else:
                ucbs=self.get_ucb(i)
                values=act_quos-cost*self.cost_rate[i]
                a=torch.argmax(ucbs*values)
                act.append(self.arms_quantile[a])
        return torch.tensor(act_quos,dtype=torch.float32)
    
    def get_ucb(self,i):
        total=self.counts[i].sum()
        constant=torch.broadcast_to(total,self.counts[i].shape)
        bonus=torch.sqrt(torch.log(constant)/self.counts[i])
        return self.values[i]+bonus
    
    def quos_to_act(self,quo,quos):
        less=quo<quos[1]
        bigger=quo>=quos[1]
        act=less*((quo-quos[1])/(quos[1]-quos[0]))
        act+=bigger*((quo-quos[0])/(quos[-1]-quos[0]))
        return act
    
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        arms=[]
        for i in range(acts.shape[0]):
            _arm=self.quos_to_act(acts[i],obs[2][i])
            arms.append(torch.argmin(torch.abs(_arm-self.arms_quantile)))
        acts=torch.tensor(arms,dtype=torch.float32)
        for i in range(acts.shape[0]):
            arms=acts[i]
            self.counts[i][arms]=self.counts[i][torch.argwhere(acts[i]==self.arms_quantile)]+1
            if rewards[1]==i:
                self.values[i][arms]=(self.counts[i][arms]/(1+self.counts[i][arms]))*self.values[i][arms]+(1/(1+self.counts[i][arms]))*1
            else:
                self.values[i][arms]=(self.counts[i][arms]/(1+self.counts[i][arms]))*self.values[i][arms]+(1/(1+self.counts[i][arms]))*0


#@ray.remote(num_cpus=1)
class LinTSMultiArmbanditAgentIn2():
    def __init__(self,arms,dimension,agents_num=[0],delta=0.5,epsilon=1/np.log(20000),R=0.01):
        self.v=[R*np.sqrt(24/epsilon*dimension*np.log(1/delta)) for i in range(arms)]
        self.B=[np.identity(dimension) for i in range(arms)]
        self.mu_hat=[np.zeros((dimension,1)) for i in range(arms)]
        self.f=[np.zeros((dimension,1)) for i in range(arms)]
        self.arms=arms
        self.arms_quantile=[0]+[1/(arms-1) for i in range(1,arms)]
        self.arms_quantile=np.array(self.arms_quantile)
        self.arms_quantile=np.cumsum(self.arms_quantile)-0.5
        self.arms_quantile=self.arms_quantile*2
        self.arms_quantile=torch.tensor(self.arms_quantile,dtype=torch.float32)
        self.last_arm=None
        self.last_context=None
        self.agents_num=agents_num
        
    def sampled(self,context,index):
        v=self.v[index]
        Bm=self.B[index]
        mu_hat=self.mu_hat[index]
        f=self.f[index]
        param1=np.matmul(context.T,mu_hat)
        param2=v**2*np.matmul(np.matmul(context.T,np.linalg.inv(Bm)),context)
        return np.random.normal(param1,param2)
    
    def quos_to_act(self,quo,quos):
        less=quo<quos[1]
        bigger=quo>=quos[1]
        act=less*((quo-quos[1])/(quos[1]-quos[0]))
        act+=bigger*((quo-quos[0])/(quos[-1]-quos[0]))
        return act
    
    def act_to_quo(self,quos,act):
        less=act<0
        bigger=act>=0
        q=less*((quos[1]-quos[0])*act+quos[1])
        q+=bigger*((quos[-1]-quos[0])*act+quos[0])
        return q
    
    def act(self,Q,Q_dist,quos,cost,info=None):
        assert info != None
        context=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(-1,1)
        temp=np.zeros((self.arms))
        for i in range(self.arms):
            temp[i]=self.sampled(context,i)
        self.last_arm=np.argmax(temp)
        #self.last_context=context
        actq=self.arms_quantile[self.last_arm].unsqueeze(0)
        return self.act_to_quo(quos,actq).unsqueeze(0)
    
    def updating(self,transition):
        obs,acts,obs_next,rewards,done=transition
        #print(transition)
        arms=[]
        _arm=self.quos_to_act(acts,obs[0:3])
        arms.append(torch.argmin(torch.abs(_arm-self.arms_quantile)))
        acts=torch.tensor(arms,dtype=torch.float32)
        rew=rewards[0]
        arm=acts[0].int()
        context=obs.numpy().reshape(-1,1)
        # print("arm",arm)
        # print("context",context)
        self.B[arm]+=np.matmul(context,context.T)
        self.f[arm]+=context*rew.numpy()
        self.mu_hat[arm]=np.matmul(np.linalg.inv(self.B[arm]),self.f[arm])
        
        
@ray.remote(num_cpus=1)
class HyperAgent():
    #choosing an agent between TraAgent and RLAgent with another bandit method
    def __init__(self,delta=0.5,epsilon=1/np.log(20000),R=0.01):
        self.TraAgent=LinTSMultiArmbanditAgentIn2(arms=9,dimension=5) 
        self.RLAgent=End2EndSACAgent(max_act=60)
        #LinTS chooser
        self.delta=delta
        self.epsilon=epsilon
        self.R=R
        
        self.ctx=np.identity(2)
        self.v=[R*np.sqrt(24/epsilon*2*np.log(1/delta)) for i in range(2)]
        self.B=[np.identity(2) for i in range(2)]
        self.mu_hat=[np.zeros((2,1)) for i in range(2)]
        self.f=[np.zeros((2,1)) for i in range(2)]
        self.last_cho=None
        
    def get_act(self,Q,Q_dist,quos,cost,info=None):
        return self.TraAgent.act(Q,Q_dist,quos,cost,info),self.RLAgent.act(Q,Q_dist,quos,cost,info)
    
    def sampled(self):
        samp=[]
        for i in range(2):
            ctx=self.ctx[i].reshape(-1,1)
            v=self.v[i]
            Bm=self.B[i]
            mu_hat=self.mu_hat[i]
            f=self.f[i]
            param1=np.matmul(ctx.T,mu_hat)
            param2=v**2*np.matmul(np.matmul(ctx.T,np.linalg.inv(Bm)),ctx)
            #print("param1:",param1,"param2:",param2,"mu_hat:",mu_hat,"Bm:",Bm,"f:",f)
            samp.append(np.random.normal(param1,param2))
        #print("samp:",samp)
        return np.argmax(samp)
    
    def act(self,Q,Q_dist,quos,cost,info=None):
        # ctxTra=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(-1,1)
        # ctxRL=np.concatenate([Q.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(1,-1)
        act1,act2=self.get_act(Q,Q_dist,quos,cost,info)
        act=[act1,act2]
        #print("act:",act)
        #self.last_cho=self.sampled()
        self.last_cho=1
        #print("cho:",self.last_cho)
        return act[self.last_cho]
    
    def update(self,tran1,tran2):
        self.TraAgent.updating(tran1)
        self.RLAgent.updating(tran2)
        
        act=self.last_cho
        rew=tran1[3][0]
        ctx=self.ctx[act].reshape(-1,1)
        self.B[act]+=np.matmul(ctx,ctx.T)
        self.f[act]+=ctx*rew.numpy()
        self.mu_hat[act]=np.matmul(np.linalg.inv(self.B[act]),self.f[act])

@ray.remote(num_cpus=1)
class HyperEPSAgent():
    #choosing an agent between TraAgent and RLAgent with another bandit method
    def __init__(self,delta=0.5,epsilon=0.05,Windows=500):
        self.TraAgent=LinTSMultiArmbanditAgentIn2(arms=9,dimension=5) 
        self.RLAgent=End2EndSACAgent(max_act=60)
        #epsilon greedy chooser
        self.epsilon=epsilon
        self.last_cho=None
        self.windows=Windows
        self.record=[[],[]]
        
    def get_act(self,Q,Q_dist,quos,cost,info=None):
        return self.TraAgent.act(Q,Q_dist,quos,cost,info),self.RLAgent.act(Q,Q_dist,quos,cost,info)
    
    def sampled(self):
        ave=np.array([np.mean(np.array(self.record[i])) if len(self.record[i])!=0 else 0 for i in range(2)])
        mam=np.argmax(ave)
        if np.random.rand()<self.epsilon:
            return np.random.randint(2)
        return mam
    
    def act(self,Q,Q_dist,quos,cost,info=None):
        # ctxTra=np.concatenate([quos.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(-1,1)
        # ctxRL=np.concatenate([Q.numpy(),cost.unsqueeze(0).numpy(),info],axis=0).reshape(1,-1)
        act1,act2=self.get_act(Q,Q_dist,quos,cost,info)
        act=[act1,act2]
        #print("act:",act)
        #self.last_cho=self.sampled()
        self.last_cho=0
        #print("cho:",self.last_cho)
        return act[self.last_cho],self.last_cho
    
    def update(self,tran1,tran2):
        self.TraAgent.updating(tran1)
        self.RLAgent.updating(tran2)
        
        act=self.last_cho
        rew=tran1[3][0]
        
        if len(self.record[act])>self.windows:
            self.record[act].pop(0)
        self.record[act].append(rew)
            
        

class HyperMultiAgent():
    def __init__(self,agents,arg_str):
        self.agents=agents
        self.arg_str=arg_str
        
    def acts(self,Q,Q_dist,quos,cost,info=None):
        handle=[]
        for i,agent in enumerate(self.agents):
            dri_nums=info['driver_nums'][i]
            handle.append(agent.act.remote(Q[i],Q_dist[i],quos[i],cost[i]*configure.cost_rate[0],dri_nums[0:len(configure.cost_rate)]))
        _actions=ray.get(handle)
        actions,info=[],[]
        for i in range(len(_actions)):
            actions.append(_actions[i][0])
            info.append(_actions[i][1])
        return torch.stack(actions),info
    
    def updating(self,transitions):
        
        for i,agent in enumerate(self.agents):
            obs,acts,obs_next,rewards,done=transitions[i]
            tran1=(obs[0],acts,obs_next[0],rewards,done)
            tran2=(obs[1],acts,obs_next[1],rewards,done)
            agent.update.remote(tran1,tran2)
            
    def log(self,writter):
        for i,agent in enumerate(self.agents):
            agent.log.remote(self.arg_str[i],writter)
            
    def obs2Context(self,Q,Q_dist,quos,cost,info=None):
        ctxs=[]
        for i in range(len(self.agents)):
            dri_nums=info['driver_nums'][i]
            context=np.array([*Q[i].numpy(),*cost[i].unsqueeze(0).numpy()*configure.cost_rate[0],dri_nums[i]]).reshape(1,-1)
            ctxs.append(torch.from_numpy(context).float())
        ctx2=torch.cat(ctxs,dim=0)
        ctxs=[]
        
        for i in range(len(self.agents)):
            dri_nums=info['driver_nums'][i]
            context=np.array([*quos[i].numpy(),*cost[i].unsqueeze(0).numpy()*configure.cost_rate[0],dri_nums[i]]).reshape(1,-1)
            ctxs.append(torch.from_numpy(context).float())
        ctx1=torch.cat(ctxs,dim=0)
        ctx=list(zip(ctx1,ctx2))
        return ctx