#python implementation of rl dsm
import numpy as np
import matplotlib.pyplot as plt
import math, time
from random import randint
import random 
import itertools
from collections import Counter

h=24 #horizon - fixed
epsilon=0 #dummy for exploration (future work)

app_data = np.array([[0.72 ,1, 24],[2.00, 2, 24],[3.15, 2, 24]])
#,[3.18, 1 ,24],[10.5, 3 ,24],[5.5, 1 ,24],[17, 3 ,24],[3.00, 1, 24],[3.00, 3, 24]])
batch_size=1; #no. of samples for offline training (1 for online) - variable

def read_data_tariff():
	print('Reading Monthly Tariff Data...')
	train_x = np.loadtxt("Tariff_data.csv", delimiter=",")
	return train_x

def weight_update(W,NW, IR, VK, Pk):
  for i in range(len(NW)):
      if i < len(NW)-1:
          error =   NW[i]-(IR[i] + NW[i+1])
      elif i == (len(NW)-1):
          error =  NW[i]- IR[i]
      W = W + error*np.matmul(np.array([VK[i]]),Pk[i])
  return(W)

#def Forward Pass
#Def Weight Updates
#Price for 31 days and 24 hours each
p_all= read_data_tariff()
print('Pricing Data =',p_all.shape)
#pi= random.randint(1, p_all.shape[0])
pi = 18
print('randomly selected day',pi)
print('Price of the selcted day',p_all[pi,:])
N=app_data.shape[0]
app_kwh = app_data[:,0]
run_time = app_data[:,1]
max_run_time = max(run_time) 
# Computing the cost Matrix
price = np.zeros((N,24))
c = p_all[pi,:]
for i in range(N):
	print("Appliance Selected =",i)
	print("runtime",run_time[i])
	x = np.ones(int(run_time[i]))
	j=0
	price_comp= []
	while j<h :
		y = app_kwh[i]*x
		p_temp = c[j:j+int(run_time[i])]
		if len(p_temp) == int(run_time[i]):
			price_comp = np.sum(np.multiply(y,p_temp))
			price[i,j] =price_comp
		j+=1
print(price.shape)

for i in range(N):
    for j in range(h):
        if price[i,j]==0:
           price[i,j] = math.inf              
print(price)          

lst = list(itertools.product([0, 1], repeat=N))
num_state = len(lst)
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
     a = price[l[0],:] 
     reward_table= np.vstack((reward_table,a))
  else:
     i=1
     a = price[l[0],:]
     while( i< len(l)):
          b= price[l[i],:] 
          a = a + b
          i+=1
     reward_table= np.vstack((reward_table,a))

'''
for l in L:
  if len(l) < 2:
     continue
  a = price[l[0],:] 
  i=1
  while( i< len(l)):
     b= price[l[i],:] 
     a = a +b
     i+=1
  price= np.vstack((price,a))
  
for i in range(price.shape[0]):
    for j in range(h):
        if price[i,j]==0:
           price[i,j] = math.inf
#price= np.vstack((price, (np.zeros((1,24)))))
price = np.insert(price, 0, (np.zeros((1,24))), axis=0)
print(price)
'''
#State Features and Action Features
state_list = [list(i) for i in itertools.product([0, 1], repeat=N)]
action_list = [list(i) for i in itertools.product([0, 1], repeat=N)]
state = np.array(state_list)
action = np.array(action_list)

#Computing the features
state_feature_n=np.zeros((state.shape[0]))
action_feature_n = np.zeros((state.shape[0]))
state_feature_m=np.zeros((state.shape[0]))
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
terminal_state = state[num_state-1]
print('Terminal State=', state[num_state-1])

#Generate the entire first episode randomly
#First episode
#random_action = 0

episode_1 = []
while True:
  random_action = np.random.choice(np.arange(1,num_state), replace=True, p=None)
  random_hour = np.random.choice(np.arange(0,h), replace=True, p=None)
  if reward_table[random_action][random_hour] != math.inf :
        state_step = init_state + action[random_action]
        break
  else:
        pass 

print( "State after taking the first action ", state_step)
x = [action[random_action] ,state_step,random_action , random_hour]
episode_1.append(x)
print(episode_1)

while True:
   if all( i==j for (i,j) in zip(state_step ,terminal_state)):  
         print("Episode has been Extracted")
         break
   else: 
         random_action_next = np.random.choice(np.arange(1,num_state), replace=False)
         random_hour_next = np.random.choice(np.arange(random_hour+1,h), replace=False)
         state_temp = state_step + action[random_action_next]
         if ((all(i <2 for i in state_temp)) and (reward_table[random_action_next][random_hour_next] != math.inf)):
               random_action =  random_action_next
               state_step = state_temp
               random_hour =  random_hour_next
               x = [action[random_action], state_step, random_action, random_hour]
               episode_1.append(x)
         else:
               pass

for episode in episode_1:
      print(episode)

#Initilization of the neural network
m0 = h+1  #number of inputs
m1 = 5*(h+1) #number of hidden units
m2=1#number of output
W1 = np.random.randn(m1,m0+1)
W2 = np.zeros((m2,m1+1),dtype=np.float128)
lamb = 1e+3
flag=1
P = np.identity(m1+1)*(1/lamb)
MSE = []
Dset_ep_all=[]
Ep_cost=[]
ep_count=0
Network_output = []
immediate_reward = []
Pk = []
VK = []
cost_esp  = []

for episode in episode_1:
        X_input = action_feature_n[episode[2]]* I[episode[3]]
        current_state = episode[1] - episode[0]
        current_af = np.dot(current_state,ec_per_hour)
        X_input = np.append(X_input,[current_af], axis=0)
        X_input = np.insert(X_input, 0, 1, axis=0) #adding 1 for bias in the begining
        #forward pass 
        output= np.dot(W2,(np.insert(np.tanh(np.dot(W1,X_input)),0,1,axis=0)))
        Network_output.append(output[0])
        immediate_reward.append(reward_table[episode[2]][episode[3]])
        V1= np.insert(np.tanh(np.dot(W1,X_input)),0,1,axis=0)
        print(V1.shape)
        vtv= np.matmul(np.transpose(np.array([V1])),np.array([V1]))
        print(vtv.shape)
        P = P - (np.matmul(np.matmul(P,vtv),P))/(1 + np.matmul(np.matmul(np.array([V1]),P),np.transpose(np.array([V1]))))
        print(P.shape)
        Pk.append(P)
        VK.append(V1)  
cost = sum(immediate_reward)
cost_esp.append(cost)
print("Length of the Episode", len(episode_1)+1)
print("Length of Network Output", len(Network_output))  
print("Length of Immediate Reward", len(immediate_reward))    
print("Network Output", Network_output)
print("Immediate Reward",immediate_reward)
W2 = weight_update(W2,Network_output, immediate_reward, VK, Pk)
#max_out_val = np.dot(W2,(np.insert(np.tanh(np.dot(W1,X_input)),0,1,axis=0)))
#Generating the episodes from now on

iteration = 1
while iteration<200:
  episode_2 = []
  P = Pk[len(Pk)-1]
  init_state = state[0]
  print('Initial State=', state[0])
  terminal_state = state[num_state -1]
  print('Terminal State=', state[num_state-1])

  Network_output_2={}
  #all actions are permissible other than "000000"
  for i in range(1,num_state):
      for j in range(0,h):  
            if (reward_table[i][j] != math.inf):
               state_temp = init_state + action[i]
               X_input = action_feature_n[i]* I[j]
               X_input = np.append(X_input,[action_feature_n[0]], axis=0) 
               X_input = np.insert(X_input, 0, 1, axis=0) #adding 1 for bias in the begining
               #forward pass
               output= np.dot(W2,(np.insert(np.tanh(np.dot(W1,X_input)),0,1,axis=0)))
               ah = [i,j]
               Network_output_2[output[0]] = ah
  
  state_step = init_state + action[Network_output_2[min(Network_output_2)][0]]
  x = [action[Network_output_2[min(Network_output_2)][0]], state_step, Network_output_2[min(Network_output_2)][0], Network_output_2[min(Network_output_2)][1]]

  episode_2.append(x)
  while True:
     if all(a==b for (a,b) in zip(state_step ,terminal_state)):
        print("Episode has been Extracted")
        break
     else:
        Network_output_2 = {}
        for i in range(1,num_state):
            for j in range(0,h):
                 state_temp = state_step + action[i]
                 if all(a<2 for a in state_temp) and (j > episode_2[len(episode_2)-1][3]) and (reward_table[i][j] != math.inf):
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
        P = P - (np.matmul(np.matmul(P,vtv),P))/(1 + np.matmul(np.matmul(np.array([V1]),P),np.transpose(np.array([V1]))))
        print(P.shape)
        Pk.append(P)
        VK.append(V1)
  W2 = weight_update(W2,Network_output, immediate_reward, VK, Pk)
  cost = sum(immediate_reward)
  cost_esp.append(cost)
  iteration+=1

print( "Number of Iterations" , iteration)
print( "Cost Matrix", len(cost_esp))
print("Final Episode")

for episode in episode_2:
    print(episode)

xbar =  np.arange(iteration)
plt.plot(xbar,cost_esp)
plt.title("Cost as a Function of Iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.savefig('Cost_fun_' + str(N) + '.pdf')
plt.close()

# Different Networks
