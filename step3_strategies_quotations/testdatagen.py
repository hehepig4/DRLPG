import QquantilePred
import Vconvlstm
import torch
import configure
import pandas as pd
import numpy as np
testdata=pd.read_hdf('new_testdata.h5')
smoothdata=pd.read_hdf('smoothing.h5')
alldata=pd.concat([testdata,smoothdata],axis=0,ignore_index=True)
alldata.reset_index(inplace=True,drop=True)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writter=SummaryWriter(log_dir='logs_gen_v/')

min_la=configure.min_la
max_la=configure.max_la
min_lo=configure.min_lo
max_lo=configure.max_lo
step_dis=step_dis=configure.step_dis

def as_tensor2(pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,weekday,minutes_in_day,trip_time_in_secs,trip_distance):
    start_block_la=int(np.round((pickup_latitude-min_la)/step_dis))
    start_block_lo=int(np.round((pickup_longitude-min_lo)/step_dis))
    end_block_la=int(np.round((dropoff_latitude-min_la)/step_dis))
    end_block_lo=int(np.round((dropoff_longitude-min_lo)/step_dis))
    matrix=np.zeros(shape=((2,round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1)),dtype=np.float32)
    matrix[0,start_block_la,start_block_lo]=1
    matrix[1,end_block_la,end_block_lo]=1
    return matrix
def transf(index,data):
    data1=data.iloc[index,:][['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','minutes_in_day','weekday','trip_time_in_secs','trip_distance']]
    matrix=as_tensor2(
        *data1.values
    ) 
    y=data.iloc[index,:][['quotation'+str(i) for i in range(10)]+['weighted_pinball_loss_Rank_loss_min','weighted_pinball_loss_Rank_loss_mean','weighted_pinball_loss_Rank_loss_max']]
    cost=data.iloc[index,:].loc['fare_amonut']
    weekday=data.iloc[index,:].loc['weekday']
    minutes_in_day=data.iloc[index,:].loc['minutes_in_day']
    end_minutes=data.iloc[index,:].loc['end_minutes']
    trip_time_in_secs=data.iloc[index,:].loc['trip_time_in_secs']
    trip_distance=data.iloc[index,:].loc['trip_distance']
    vector_angle=np.arctan2(data.iloc[index,:].loc['dropoff_latitude']-data.iloc[index,:].loc['pickup_latitude'],data.iloc[index,:].loc['dropoff_longitude']-data.iloc[index,:].loc['pickup_longitude'])
    other_quos=y[['quotation'+str(i) for i in range(10)]].values
    predic_quos=y[['weighted_pinball_loss_Rank_loss_min','weighted_pinball_loss_Rank_loss_mean','weighted_pinball_loss_Rank_loss_max']].values
    end_abs_minutes=int(data.iloc[index,:].loc['abs_end_minutes'].item())
    start_abs_minutes=int(data.iloc[index,:].loc['abs_minutes'].item())
    #[start_block,end_block],feas,predic,other_quos,cost
    return matrix,[weekday,minutes_in_day,end_minutes,trip_time_in_secs,trip_distance,vector_angle,end_abs_minutes,start_abs_minutes],predic_quos,other_quos,cost
def sort(data):
    data.sort_values(by=['abs_end_minutes'],inplace=True,ascending=False)
sort(alldata)
P_handle={i:np.zeros(shape=(81,131)) for i in reversed(range(2868)) }
Z_handle={i:np.zeros(shape=(81,131)) for i in reversed(range(2868)) }
Q_handle={i:np.zeros(shape=(2,81,131)) for i in reversed(range(2868)) }
count={i:np.ones(shape=(81,131)) for i in reversed(range(2868)) }
all_index=len(alldata)
gamma=0.99
# 1326-2867
from tqdm import tqdm
time_count=2867
with tqdm(total=all_index) as pbar:
    for i in range(all_index):
        mat,fea,pred,quos,cost=transf(i,testdata)
        pbar.update(1)
        end_time=fea[-2]
        start_time=fea[-1]
        while end_time<time_count:
            time_count-=1
            P_handle[time_count]+=Z_handle[time_count+1]*gamma
            Z_handle[time_count]=P_handle[time_count]/count[time_count]
            tqdm.write(str(time_count)+' '+str(np.sum(Z_handle[time_count])))
            writter.add_scalar('Z_handle',np.sum(Z_handle[time_count]),17262-time_count)
            writter.add_scalar('Q_handle_start',np.sum(Q_handle[time_count+1][0,:,:]),17262-time_count)
            writter.add_scalar('Q_handle_end',np.sum(Q_handle[time_count+1][1,:,:]),17262-time_count)
        q=quos[0]
        end_block=(np.nonzero(mat[1])[0].item(),np.nonzero(mat[1])[1].item())
        Q_handle[time_count][1][end_block]+=1
        start_block=(np.nonzero(mat[0])[0].item(),np.nonzero(mat[0])[1].item())
        Q_handle[time_count][0][start_block]+=1
        P_handle[start_time][start_block]+=(Z_handle[end_time][end_block]+q)*np.power(gamma,end_time-start_time)
        count[start_time][start_block]+=1
def get_quans(start_block_z,end_block_z,t_start,t_end,q):
    return Z_handle[t_start][start_block_z]-np.power(gamma,t_start-t_end)*(Z_handle[t_end][end_block_z]+q)
t=max(list(Z_handle.keys()))
all_t=range(t)
tenz=[]
tenq=[]
all_ten=[]
import torch
for i in reversed(all_t):
    tenz.append(torch.from_numpy(Z_handle[i]).unsqueeze(0))
    tenq.append(torch.from_numpy(Q_handle[i]))
    all_ten.append(torch.cat([tenz[-1],tenq[-1]],dim=0))
test_ten=torch.stack(all_ten)
test_ten=test_ten[1326:]
torch.save(test_ten,'test_ten.pt')