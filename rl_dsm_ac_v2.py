#python implementation of rl dsm
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import math, time
from random import randint
import random 
import itertools
from collections import Counter

h=24 #horizon - fixed
epsilon=0 #dummy for exploration (future work)

app_data = np.array([[0.72 ,1, 24],[2.00, 2, 24],[3.15, 3, 24]])
#,[3.18, 1 ,24],[10.5, 3 ,24],[5.5, 1 ,24],[17, 3 ,24],[3.00, 1, 24],[3.00, 3, 24]])
batch_size=1; #no. of samples for offline training (1 for online) - variable
episode =300; #no. of iteraions for weight update - variable

def read_data_tariff():
	print('Reading Monthly Tariff Data...')
	train_x = np.loadtxt("Tariff_data.csv", delimiter=",")
	return train_x

#FUNCTION FOR FEATURE GENERATION AND OUTPUT
def nn_eval(x):
        state_feature = np.dot(x[1] - x[0],ec_per_hour)
        nn_input = np.append(action_feature_n[x[2]]* I[x[3]],[state_feature], axis=0)
        mn =  np.mean(nn_input)
        nn_input = nn_input - mn
        nn_input = np.insert(nn_input, 0, 1, axis=0) #adding 1 for bias in the begining                
        nn_output= np.dot(W2,(np.insert(np.tanh(np.dot(W1,nn_input)),0,1,axis=0)))
        #return{'nn input feat':nn_input, 'nn output':nn_output}
        return(x,nn_input, nn_output)

#FUNCTION FOR ERROR COMPUTATION

#FUNCTION FOR WEIGHT UPDATE
def nn_update(x_in,error):
    global P
    V1= np.insert(np.tanh(np.dot(W1,x_in)),0,1,axis=0)
    vtv= np.matmul(np.transpose(np.array([V1])),np.array([V1]))
    P = P - (np.matmul(np.matmul(P,vtv),P))/1 + np.matmul(np.matmul(np.array([V1]),P),np.transpose(np.array([V1])))
    delW=error*np.matmul(V1,P)
    return(delW)    
  

#FUNCTION FOR ELECTING RANDOM ACTIONS

#def Forward Pass
#Def Weight Updates
#cost for 31 days and 24 hours each
#Price for 31 days and 24 hours each
p_all= read_data_tariff()
print('Pricing Data =',p_all.shape)
#pi= random.randint(1, p_all.shape[0])
pi = 18
print('randomly selected day',pi)
print('Price of the selcted day',p_all[pi,:])

p = p_all[pi,:]
#np.array([35.33,31.36,32.27,32.35,30.80,33.87,43.19,48.24,43.47,42.13,39.28,37.35,34.77,33.20,31.39,31.54, 35.84,47.29,45.17,39.98,35.65,34.07,34.32,32.66])

N=app_data.shape[0]
app_kwh = app_data[:,0]
run_time = app_data[:,1]
max_run_time = max(run_time) 

# Computing the cost Matrix
cost = np.zeros((N,24))
for i in range(N):
	print("Appliance Selected =",i)
	print("runtime",run_time[i])
	x = np.ones(int(run_time[i]))
	j=0
	cost_comp= []
	while j<h :
		y = app_kwh[i]*x
		p_temp = p[j:j+int(run_time[i])]
		if len(p_temp) == int(run_time[i]):
			cost_comp = np.sum(np.multiply(y,p_temp))
			cost[i,j] =cost_comp
		j+=1
print(cost.shape)

for i in range(N):
    for j in range(h):
        if cost[i,j]==0:
           cost[i,j] = float('inf')
print(cost)          

lst = list(itertools.product([0, 1], repeat=N))
state_num = len(lst)
N_a=np.arange(N)
L =[]
for i in range(len(lst)):
     x=[j for j, e in enumerate(lst[i]) if e != 0]
     L.append(x)
print(L)
print(len(L))
reward_table = np.zeros((1,h))

for l in L:
  if len(l) == 0:
     continue
  elif len(l) == 1:  
     a = cost[l[0],:] 
     reward_table= np.vstack((reward_table,a))
  else:
     i=1
     a = cost[l[0],:]
     while( i< len(l)):
          b= cost[l[i],:] 
          a = a + b
          i+=1
     reward_table= np.vstack((reward_table,a))

#State Features and Action Features
state_list = [list(i) for i in itertools.product([0, 1], repeat=N)]
action_list = [list(i) for i in itertools.product([0, 1], repeat=N)]
state = np.array(state_list)
action = np.array(action_list)

#Computing the features
state_feature_n = np.zeros((state.shape[0]))
action_feature_n = np.zeros((state.shape[0]))
state_feature_m = np.zeros((state.shape[0]))
action_feature_m = np.zeros((state.shape[0]))

ec_per_hour = [a*b for a,b in zip(run_time,app_kwh)]
for i in range(state.shape[0]):
   state_feature_n[i] = np.dot(state[i],ec_per_hour)
   action_feature_n[i] = np.dot(action[i],ec_per_hour)
   state_feature_m[i] = np.dot(state[i],app_kwh)
   action_feature_m[i] = np.dot(action[i],app_kwh)
#Create an Indentity
I = np.identity(24)
init_state = state[0]
print('Initial State=', state[0])
terminal_state = state[state_num-1]
print('Terminal State=', state[state_num-1])

#Initilization of the neural network
m0 = h+1  #number of inputs
m1 = 500 #number of hidden units
m2=1#number of output
W1 = np.random.randn(m1,m0+1)
W2 = np.zeros((m2,m1+1),dtype=np.float128)
lamb = 1e-4
flag=1
P = np.identity(m1+1)*(1/lamb)
MSE = []
Dset_ep_all=[]
Ep_cost=[]
ep_count=0
Network_output = []
immediate_reward = []
cost_esp  = []

#Generate the entire first episode randomly
#First episode
#random_action = 0
episode = []
while True:
  random_action = np.random.choice(np.arange(1,state_num), replace=True, p=None)
  random_hour = np.random.choice(np.arange(0,h), replace=True, p=None)
  if reward_table[random_action][random_hour] != float('inf') :
        state_step = init_state + action[random_action]
        immediate_reward=reward_table[random_action][random_hour]
        break
  else:
        pass 

print( "State after taking the first action ", state_step)
step = [action[random_action] ,state_step,random_action , random_hour, immediate_reward]
episode.append(nn_eval(step))
print(episode)

while True:
   if all( i==j for (i,j) in zip(state_step ,terminal_state)):  
         print("Episode has been Extracted")
         break
   else: 
         random_action_next = np.random.choice(np.arange(1,state_num), replace=False)
         random_hour_next = np.random.choice(np.arange(random_hour+1,h), replace=False)
         state_temp = state_step + action[random_action_next]
         if ((all(i <2 for i in state_temp)) and (reward_table[random_action_next][random_hour_next] != float('inf'))):
               random_action =  random_action_next
               state_step = state_temp
               random_hour =  random_hour_next
               immediate_reward=reward_table[random_action][random_hour]
               step = [action[random_action], state_step, random_action, random_hour, immediate_reward]
               episode.append(nn_eval(step))
         else:
               pass

for step in episode:
    print(step)

for s in range(len(episode)):
    if s < len(episode)-1:
        error =   episode[s][2]-(episode[s][0][4]+ episode[s+1][2])
    elif s == len(episode)-1:
        error =   episode[s][2]-episode[s][0][4]
    W2=W2+nn_update(episode[s][1],error)    

#Generating episodes via NN
max_iter=500
dataset=[]
episode_summary=[]
for iteration in range(max_iter):
    episode = []
    #P = Pk[len(Pk)-1]
    state_current = state[0]
    state_terminal = state[state_num-1]
    hi=0
    while True:
        if all(a==b for (a,b) in zip(state_current ,terminal_state)):
            print("Terminal State encountered, episode generated")
            break
        else:
            nn_out_all={}
            for i in range(1,len(action)):
                state_next = state_current + action[i]  
                if all(a<2 for a in state_next):
                    hmax=int(h-max((state_terminal-state_current)*(run_time)))
                    for j in range(hi,hmax+1):  
                        step = [action[i], state_next, i, j]
                        output=nn_eval(step)[2]
                        mapz = [i,j]
                        nn_out_all[output[0]] = mapz
            min_action=nn_out_all[min(nn_out_all)][0]
            state_next = state_current + action[min_action]
            min_hour=nn_out_all[min(nn_out_all)][1]
            print(min_action, min_hour)
	    
            immediate_reward=reward_table[min_action][min_hour]
            step = [action[min_action], state_next, min_action, min_hour, immediate_reward]
            episode.append(nn_eval(step))            
            state_current=state_next
            hi=min_hour+1
            nn_out_all
    
    cost=0    
    for s in range(len(episode)):
        cost+=(episode[s][0][4])
        if s < len(episode)-1:
            error = episode[s][2]-(episode[s][0][4]+ episode[s+1][2])
        elif s == len(episode)-1:
            error = episode[s][2]-episode[s][0][4]
        W2=W2+nn_update(episode[s][1],error)  
        print("W2 updated")
    
    dataset.append(episode)
    episode_summary.append([iteration,cost])
    print(episode_summary)
 
    dataset.append(episode)
    #print("This is episode no: ",iteration)
    #print (dataset[0])


'''    
    while True:
       if all(a==b for (a,b) in zip(state_step ,terminal_state)):
             print("Episode has been Extracted")
             break
       else:
             Network_output_2 = {}
             for i in range(1,state_num):
               for j in range(0,h):
                 state_temp = state_step + action[i]
                 if all(a<2 for a in state_temp) and (j > episode_2[len(episode_2)-1][3]):
                             X_input = action_feature_n[i]* I[j]
                             current_state = episode_2[len(episode_2)-1][1] - episode_2[len(episode_2)-1][0]
                             current_af = np.dot(current_state,ec_per_hour)
                             X_input = np.append(X_input,[current_af], axis=0)
                             X_input = np.insert(X_input, 0, 1, axis=0) #adding 1 for bias in the begining
                             #forward pass
                             output= np.dot(W2,(np.insert(np.tanh(np.dot(W1,X_input)),0,1,axis=0)))                          
                             ah = [i,j]
                             Network_output_2[output[0]] = ah
                 else:
                             pass
        
       state_step = state_step + action[Network_output_2[min(Network_output_2)][0]]
       x = [action[Network_output_2[min(Network_output_2)][0]], state_step, Network_output_2[min(Network_output_2)][0], Network_output_2[min(Network_output_2)][1]]
       episode_2.append(x)

    for episode in episode_2:
           print(episode)
    Pk = []
    VK = []
    Network_output = []
    immediate_reward = []
    for episode in episode_2:
        X_input = action_feature_n[episode[2]]* I[episode[3]]
        current_state = episode[1] - episode[0]
        current_af = np.dot(current_state,ec_per_hour)
        X_input = np.append(X_input,[current_af], axis=0)
        X_input = np.insert(X_input, 0, 1, axis=0) #adding 1 for bias in the begining
        #forward pass
        #First Layer 
        output= np.dot(W2,(np.insert(np.tanh(np.dot(W1,X_input)),0,1,axis=0)))
        Network_output.append(output[0])
        immediate_reward.append(reward_table[episode[2]][episode[3]])
        V1= np.insert(np.tanh(np.dot(W1,X_input)),0,1,axis=0)
        vtv= np.matmul(np.transpose(np.array([V1])),np.array([V1]))
        print(vtv.shape)
        P = P - (np.matmul(np.matmul(P,vtv),P))/1 + np.matmul(np.matmul(np.array([V1]),P),np.transpose(np.array([V1])))
        print(P.shape)
        Pk.append(P)
        VK.append(V1)
    W2 = weight_update(W2,Network_output, immediate_reward, VK, Pk)
    cost = sum(immediate_reward)
    cost_esp.append(cost)
    iteration+=1
'''
episode_summary

xbar =  episode_summary[:][0]
cost_esp = episode_summary[:][1]
plt.plot(xbar,cost_esp)
plt.title("Cost as a Function of Iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.savefig('Cost_fun_' + str(N) + '.pdf')
plt.close()

