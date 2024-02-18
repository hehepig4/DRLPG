from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import re
import os
import numpy as np
from matplotlib import pyplot as plt

names=[
    # 'random',
    #'LinTS',
    # 'greedy',
    # 'Tom',
    # 'ucb',
    # 'D4VQN',
    # 'D4EQN',
    # 'D4QN'
    "FIXRL_trEP0EPSHyper",
    "FIXRL_trEP1EPSHyper",
    "FIXRL_trEP2EPSHyper",
    "FIXTra_trEP0EPSHyper",
    "FIXTra_trEP1EPSHyper",
    "FIXTra_trEP2EPSHyper",
    "EPS005_trEP0EPSHyper",
    "EPS005_trEP1EPSHyper",
    "EPS005_trEP2EPSHyper",
]

dirs=['newplogs/re_high_cost_1_'+names[i]+'/' for i in range(len(names))]
#epochs=[('epo'+str(i)) for i in range(1)]
base=[10093,17691,29234,8352,14092,21877,8432,15439,26793]
#base=[1484,2665,4391,1170,1937,3042,1207,2241,3856]
def res(num):
    ans_dict={names[num]:{}}
    subdirs=os.listdir(dirs[num])
    for subdir in subdirs:
        counter=0
        ave=0
        files=os.listdir(dirs[num]+subdir)
        for file in files:
            if re.match('epo',file):
                counter+=1
                data=pd.read_csv(dirs[num]+subdir+'/'+file)
                ave+=data['value'].iloc[-1]
        ave=ave/counter
        ans_dict[names[num]].update({subdir:ave})
    return ans_dict
envs=[
    '0 0.05 60 120',
    '0 0.1 60 120',
    '0 0.2 60 120',
    '3 0.05 60 120',
    '3 0.1 60 120',
    '3 0.2 60 120',
    '5 0.05 60 120',
    '5 0.1 60 120',
    '5 0.2 60 120',
]
# envs=[
#     '0 0.05 500 560',
#     '0 0.1 500 560',
#     '0 0.2 500 560',
#     '3 0.05 500 560',
#     '3 0.1 500 560',
#     '3 0.2 500 560',
#     '5 0.05 500 560',
#     '5 0.1 500 560',
#     '5 0.2 500 560',
# ]

if __name__ == '__main__':
    all_dict={}
    for i in range(len(names)):
        all_dict.update(res(i))
    total = []
    for env in envs:
        #print('______________')
        #print(env,':',base[envs.index(env)])
        tmp = {}
        for name in names:
            value=np.round(all_dict[name][env]/base[envs.index(env)]*100-100,1)
            tmp[name]=value
        total.append(tmp)
    for name in names:
        ave=0
        for i in range(len(total)):
            ave+=total[i][name]
        print(ave/9)
        
# args=event_accumulator.EventAccumulator('heurlogs/greedy/epo0/cum_reward_cum_rewardmutLinTS0 0.1 60 120/events.out.tfevents.1663309481.amax.1183232.2')
# event=args.Reload()
# pd=pd.DataFrame(event.scalars.Items('cum_reward'))
# pd.to_csv('temp.csv')
# pass