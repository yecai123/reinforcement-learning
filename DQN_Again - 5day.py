import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ps = 1 #state 和 action 数据的粒度为1
data = pd.read_csv("data2.csv")
class MG:
    def __init__(self):
        ac=[]                           #action space [MT,VE]
        for i in range(0,61,ps):
            for j in range(0,61,ps):
                if i==0 or i>30:
                    ac.append([i,j])

        state=[]                        #state space [t,SOC]
        for i in range(24):
            for j in range(101):
                #state=np.append(state,[i+1,j],axis=0)
                state.append([i,j])

        self.action_space = ac
        self.state_space = state
        self.n_actions = len(ac)
        self.n_features = 2
        self.ve = 0

    def reset(self):
        n = pd.Series(data.iloc[0],dtype=float)                 #引入环境的原始数据
        ob = [0,int(n[1]),int(n[2]),int(n[3]),20,int(n[4])]      #ob = [t=0,ELD,HLD,CLD,SOC=20,PV]
        ob=np.array(ob)
        return ob

    def step(self,action,ob):                                 #实施动作+返回奖励 action=[MT,VE] ob=[t,ELD,HLD,CLD,SOC,PV]   BP为正时视为电池放电 VE为正时视为向电网购电
        mt = self.action_space[action][0]
        ve = self.action_space[action][1]
        [t,eld,hld,cld,soc,pv] = ob
        vacancy,fine,self.cost = 0,0,0
        
        
        t_ = ob[0]+1
        n = pd.Series(data.iloc[t_%24])
        eld_ = int(n[1])
        hld_ = int(n[2])
        cld_ = int(n[3])
        pv_ = int(n[4])
        tou_ = n[5]
        bp = eld_ + cld_*0.25 - mt - ve - pv
        soc_ = int(soc - bp/4)

        if (soc_>=90 and bp<0) or (soc_>90):                     #对电量过充的处理，过充的电量折算到VE中
            vacancy = eld_ + cld_*0.25 - mt - ve - pv_ - bp + (360-soc_*4)     # 过充，多余电量折算为-ve 实际守恒VE=ELD+0.25CLD-MT-PV-BP
            soc_=90
            fine = -10*(abs(bp)+abs(vacancy))
        elif (soc_<=20 and bp>0) or (soc_<20):
            vacancy = eld_ + cld_*0.25 - mt - ve - pv_ - bp + (80-soc_*4)    # 过放，亏损电量折算为ve 实际守恒VE=ELD-MT-PV+BP
            soc_=20
            fine = -10*(abs(bp)+abs(vacancy))
        else: 
            vacancy = eld_ + cld_*0.25 - mt - ve - pv_ - bp
            fine = -10*(abs(vacancy))
        #print('va = ',vacancy)
        #print('fine = ',fine)
        ob_ = [t_,eld_,hld_,cld_,soc_,pv_]                                               #观测值[t,ELD,HLD,CLD,SOC,PV]

        if (hld_ - 1.533*mt) / 0.81 < 0:
            f_sb = 0
        else:
            f_sb = ((hld_ - 1.533*mt) / 0.81)
        f_mt = mt/0.3

        reward = fine - (tou_ * ve + (f_mt + f_sb) * 0.349)      #reward 为运营成本，由购电成本和燃料成本构成
        self.cost = (tou_ * ve + (mt/0.3 + ((hld_ - 1.533*mt) / 0.81)) * 0.349)
        #[mt,ve,pv,bp,ld,soc,vacancy]
        self.pb=[mt,ve,pv_,bp,eld_+0.25*cld_,soc_,vacancy,fine,reward-fine]
        self.hld = hld_
        if flag ==24:
            t_end = 24
        else:
            t_end = 120
                    
        if t_ == t_end:
            done = True
        else :
            done = False
        
        return ob_, reward, done




np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        
###################################################################################
        self.units = 50

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------  2 layers n_l1 → n_action units
        
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):  
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.units, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
                l2 = tf.matmul(l1, w2) + b2

            # 3

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, n_l1], initializer=b_initializer, collections=c_names)
                l3 = tf.matmul(l2, w3) + b3
                
            # 4

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, n_l1], initializer=b_initializer, collections=c_names)
                l4 = tf.matmul(l3, w4) + b4

            # 5
            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l4, w5) + b5


        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
                l2 = tf.matmul(l1, w2) + b2

            # 3
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, n_l1], initializer=b_initializer, collections=c_names)
                l3 = tf.matmul(l2, w3) + b3

            # 4
            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, n_l1], initializer=b_initializer, collections=c_names)
                l4 = tf.matmul(l3, w4) + b4

            # 5
            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l4, w5) + b5

    def store_transition(self, s, a, r, s_):
        s = np.array([s[2],s[4]])
        s_ = np.array([s_[2],s_[4]])

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = np.array([observation[2],observation[4]])[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()





env = MG()
#n_actions,n_features 分别是Qtable列数（A）、行数（S）
RL = DeepQNetwork(env.n_actions, env.n_features,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=5000,
                  # output_graph=True
                  )

flag = 24
re_max_l = []
re_max = -10000000
re_mean_l = []
re_mean = 0
re_meann = 0
action_op = []
step = 0
for episode in range(5000):
    # initial observation
    observation = env.reset()
    re,cost = 0,0
    action_list = []

    while True:
        # RL choose action based on observation
        action = RL.choose_action((observation))
        
        action_list.append(action)
        # RL take action and get next observation and reward
        # action=[MT,BP] ob=[t,ELD,HLD,CLD,SOC,PV]
        observation_, reward, done = env.step(action,observation)
        re = re + reward
        cost += env.cost
        
        RL.store_transition(observation, action, reward, observation_)
        if (step > 200) and (step % 5 == 0):
            RL.learn()
        
        # swap observation
        observation = observation_
        # break while loop when end of this episode
        if done:
            break
        step += 1

    re_meann += re
    if re > re_max:
        re_max = re
        action_op = action_list
        count = 0
    else:
        count += 1
    
    if episode % 100 == 0:
        if episode == 0:
            re_mean = re_max
        else:
            re_mean = re_meann/100
            re_meann = 0
    if episode % 1000 == 0:
        print(episode,'re_max =',int(re_max),'re_mean =',int(re_mean))
    re_max_l.append(re_max)
    re_mean_l.append(re_mean) 
    if episode % 1000 == 0:
        plt.plot(re_max_l)
        plt.plot(re_mean_l)
        plt.show()
        epi = episode
#    if count == 30000:
#        epi = episode
#        break
# end of game,Q-table learned
plt.plot(re_max_l,label='re_max')
plt.plot(re_mean_l,label='re_mean')
pd.DataFrame([re_max_l,re_mean_l]).to_csv("re_record.csv")
print('re_max = ',re_max)
plt.xlabel('Episode')
plt.ylabel('Reward (Yuan)')
plt.legend(loc='lower right')
re_fig = plt.gcf()
plt.show()


#################### result based on the learned Q-learning ###################


observation = env.reset()
ac = env.action_space
ac_list,ob_list,pb,hb = [],[],[],[]
re,cost = 0,0
ve_ = 0
       #action space [MT,BP]

for i in range(24):
    action = action_op[i]
    ac_list.append(ac[action])
    ob_list.append(observation)
    observation_, reward, done = env.step(action,observation)
    pb.append(env.pb)
    mt_ = env.pb[1]
    hld_ = env.hld
    hb.append([1.533*mt_,hld_-1.533*hld_,hld_])
    observation = observation_
    re += reward
    cost += env.cost
    
print('re = ',re)
print('cost = ',cost)
pdt = pd.DataFrame(pb)
pdt.columns = ['mt','ve','pv','bp','ld','soc','vacancy','fine','cost']
pdt.to_csv("pb.csv")


############################## Demonstration ##############################


resultdata = pd.read_csv('pb.csv')
mt = np.array(pd.Series(resultdata['mt']))
ve = np.array(pd.Series(resultdata['ve']))
bp = np.array(pd.Series(resultdata['bp']))
pv = np.array(pd.Series(resultdata['pv']))
ld = np.array(pd.Series(resultdata['ld']))
va = np.array(pd.Series(resultdata['vacancy']))
bpp=[]
bpm=[]
for i in range(24):
    if bp[i]>0:
        bpp.append(bp[i])
        bpm.append(0)
    else:
        bpm.append(bp[i])
        bpp.append(0)
t = np.arange(24)
plt.plot(ld,linewidth=2,color='black',marker='_',markersize=10)
p1 = plt.bar(t,mt,width=0.92,label='MT')
p2 = plt.bar(t,ve,width=0.92,bottom=mt,label='GP')
p3 = plt.bar(t,pv,width=0.92,bottom=mt+ve,label='PV')
p4 = plt.bar(t,bpp,width=0.92,bottom=mt+ve+pv,color='red',label='BD')
p0 = plt.bar(t,bpm,width=0.92,label='BC')
plt.legend(loc='upper right')
plt.xlabel('Time (h)')
plt.ylabel('Power Output (kWh)')
op_fig = plt.gcf()
plt.show()        
    

#################### result reop ###################
cost = 0
tou = np.array(pd.Series(data['TOU']))
hld = np.array(pd.Series(data['HLD']))
for i in range(24):
    if (va[i]<0) and (va[i]+ve[i]>=0):
        ve[i] = va[i] + ve[i]
        va[i] = 0
    #电能输出不足 增加电网购电
    if va[i]>0:
        ve[i] = ve[i] + va[i]
        va[i] = 0
    cost += (tou[i] * ve[i] + (mt[i]/0.3 + ((hld[i] - 1.533*mt[i]) / 0.81)) * 0.349)
print('reop_cost = ',cost)
resultdata['ve'] = ve
resultdata['vacancy'] = va

mt = np.array(pd.Series(resultdata['mt']))
ve = np.array(pd.Series(resultdata['ve']))
bp = np.array(pd.Series(resultdata['bp']))
pv = np.array(pd.Series(resultdata['pv']))
ld = np.array(pd.Series(resultdata['ld']))
soc = np.array(pd.Series(resultdata['soc']))
va = np.array(pd.Series(resultdata['vacancy']))
bpp=[]
bpm=[]
zeros=np.zeros(24)
for i in range(24):
    if bp[i]>0:
        bpp.append(bp[i])
        bpm.append(0)
    else:
        bpm.append(bp[i])
        bpp.append(0)
t = np.arange(24)

fig3 = plt.figure()
ax1 = fig3.add_subplot(111)
ax1.plot(ld-bpm,linewidth=2,color='black',marker='_',label='LD',markersize=10)
ax1.plot(zeros-bpm,linewidth=2,color='brown',label='BC')
#ax1.plot(soc,linewidth=2,color='navy',marker='*',label='SOC',markersize=10)

#plt.bar(t,-ld,width=0.92,color='gray')
p1 = ax1.bar(t,mt,width=0.92,label='MT')
p2 = ax1.bar(t,ve,width=0.92,bottom=mt,label='GP')
p3 = ax1.bar(t,bpp,width=0.92,bottom=mt+ve,color='red',label='BD')
p4 = ax1.bar(t,pv,width=0.92,bottom=mt+ve+bpp,label='PV+WT')
#p0 = plt.bar(t,bpm,width=0.92,color='brown',label='BC')
ax1.legend(loc='upper left')
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Power Output (kWh)')

ax2 = ax1.twinx()
plt.ylim((0,100))
ax2.plot(soc,linewidth=2,color='navy',marker='*',label='SOC',markersize=10)
ax2.set_ylabel('State of charge')
ax2.legend(loc='upper right')

############################# save result ####################################

foldername = 'DQN-S0'+' e='+str(epi)+' re='+str(int(re))+' cost='+str(int(cost))
if os.path.exists(foldername)==False:
    os.makedirs(foldername)
cwd = os.getcwd()
pdt.to_csv(cwd+'\\'+foldername+"\\pb.csv")
pd.DataFrame([re_max_l,re_mean_l]).to_csv(cwd+'\\'+foldername+"\\re_record.csv")
op_fig.savefig(cwd+'\\'+foldername+"\\op_fig.png",dpi=100)
re_fig.savefig(cwd+'\\'+foldername+"\\re_fig.png",dpi=100)

reop_fig = plt.gcf()
plt.show()
reop_fig.savefig(cwd+'\\'+foldername+"\\reop_fig.png",dpi=100)




############################## Again ###################################!!!!!!!!!

starttime = datetime.datetime.now()
env = MG()
#n_actions,n_features 分别是Qtable列数（A）、行数（S）
#RL2 = DeepQNetwork2(env.n_actions, env.n_features,
#                  learning_rate=0.01,
#                  reward_decay=0.9,
#                  e_greedy=0.9,
#                  replace_target_iter=200,
#                  memory_size=5000,
#                  # output_graph=True
#                  )
data = pd.read_csv("data - 5day.csv")
flag = 120
re_max_l = []
re_max = -10000000
re_mean_l = []
re_mean = 0
re_meann = 0
action_op = []
step = 0
for episode in range(1500):
    # initial observation
    observation = env.reset()
    re,cost = 0,0
    action_list = []

    while True:
        # RL choose action based on observation
        action = RL.choose_action((observation))
        
        action_list.append(action)
        # RL take action and get next observation and reward
        # action=[MT,BP] ob=[t,ELD,HLD,CLD,SOC,PV]
        observation_, reward, done = env.step(action,observation)
        re = re + reward
        cost += env.cost
        
        RL.store_transition(observation, action, reward, observation_)
        if (step > 200) and (step % 5 == 0):
            RL.learn()
        
        # swap observation
        observation = observation_
        # break while loop when end of this episode
        if done:
            break
        step += 1

    re_meann += re
    if re > re_max:
        re_max = re
        action_op = action_list
        count = 0
    else:
        count += 1
    
    if episode % 100 == 0:
        if episode == 0:
            re_mean = re_max
        else:
            re_mean = re_meann/100
            re_meann = 0
    if episode % 1000 == 0:
        print(episode,'re_max =',int(re_max),'re_mean =',int(re_mean))
    re_max_l.append(re_max)
    re_mean_l.append(re_mean) 
    if episode % 1000 == 0:
        plt.plot(re_max_l)
        plt.plot(re_mean_l)
        plt.show()
        epi = episode
#    if count == 30000:
#        epi = episode
#        break
# end of game,Q-table learned
plt.plot(re_max_l,label='re_max')
plt.plot(re_mean_l,label='re_mean')
pd.DataFrame([re_max_l,re_mean_l]).to_csv("re_record.csv")
print('re_max = ',re_max)
plt.xlabel('Episode')
plt.ylabel('Reward (Yuan)')
plt.legend(loc='lower right')
re_fig = plt.gcf()
plt.show()
endtime = datetime.datetime.now()
print (endtime - starttime)
#################### result based on the learned Q-learning ###################


observation = env.reset()
ac = env.action_space
ac_list,ob_list,pb,hb = [],[],[],[]
re,cost = 0,0
ve_ = 0
       #action space [MT,BP]

for i in range(120):
    action = action_op[i]
    ac_list.append(ac[action])
    ob_list.append(observation)
    observation_, reward, done = env.step(action,observation)
    pb.append(env.pb)
    mt_ = env.pb[1]
    hld_ = env.hld
    hb.append([1.533*mt_,hld_-1.533*hld_,hld_])
    observation = observation_
    re += reward
    cost += env.cost
    
print('re = ',re)
print('cost = ',cost)
pdt = pd.DataFrame(pb)
pdt.columns = ['mt','ve','pv','bp','ld','soc','vacancy','fine','cost']
pdt.to_csv("pb.csv")


############################## Demonstration ##############################


resultdata = pd.read_csv('pb.csv')
mt = np.array(pd.Series(resultdata['mt']))
ve = np.array(pd.Series(resultdata['ve']))
bp = np.array(pd.Series(resultdata['bp']))
pv = np.array(pd.Series(resultdata['pv']))
ld = np.array(pd.Series(resultdata['ld']))
va = np.array(pd.Series(resultdata['vacancy']))
bpp=[]
bpm=[]
for i in range(120):
    if bp[i]>0:
        bpp.append(bp[i])
        bpm.append(0)
    else:
        bpm.append(bp[i])
        bpp.append(0)
t = np.arange(120)
plt.plot(ld,linewidth=2,color='black',marker='_',markersize=10)
p1 = plt.bar(t,mt,width=0.92,label='MT')
p2 = plt.bar(t,ve,width=0.92,bottom=mt,label='GP')
p3 = plt.bar(t,pv,width=0.92,bottom=mt+ve,label='PV')
p4 = plt.bar(t,bpp,width=0.92,bottom=mt+ve+pv,color='red',label='BD')
p0 = plt.bar(t,bpm,width=0.92,label='BC')
plt.legend(loc='upper right')
plt.xlabel('Time (h)')
plt.ylabel('Power Output (kWh)')
op_fig = plt.gcf()
plt.show()        
    

#################### result reop ###################
cost = 0
tou = np.array(pd.Series(data['TOU']))
hld = np.array(pd.Series(data['HLD']))
for i in range(120):
    if (va[i]<0) and (va[i]+ve[i]>=0):
        ve[i] = va[i] + ve[i]
        va[i] = 0
    #电能输出不足 增加电网购电
    if va[i]>0:
        ve[i] = ve[i] + va[i]
        va[i] = 0
    cost += (tou[i] * ve[i] + (mt[i]/0.3 + ((hld[i] - 1.533*mt[i]) / 0.81)) * 0.349)
print('reop_cost = ',cost)
resultdata['ve'] = ve
resultdata['vacancy'] = va

mt = np.array(pd.Series(resultdata['mt']))
ve = np.array(pd.Series(resultdata['ve']))
bp = np.array(pd.Series(resultdata['bp']))
pv = np.array(pd.Series(resultdata['pv']))
ld = np.array(pd.Series(resultdata['ld']))
soc = np.array(pd.Series(resultdata['soc']))
va = np.array(pd.Series(resultdata['vacancy']))
bpp=[]
bpm=[]
zeros=np.zeros(120)
for i in range(120):
    if bp[i]>0:
        bpp.append(bp[i])
        bpm.append(0)
    else:
        bpm.append(bp[i])
        bpp.append(0)
t = np.arange(120)

fig3 = plt.figure(figsize=(15,6))
ax1 = fig3.add_subplot(111)
ax1.plot(ld-bpm,linewidth=2,color='black',marker='_',label='LD',markersize=10)
ax1.plot(zeros-bpm,linewidth=2,color='brown',label='BC')
#ax1.plot(soc,linewidth=2,color='navy',marker='*',label='SOC',markersize=10)

#plt.bar(t,-ld,width=0.92,color='gray')
p1 = ax1.bar(t,mt,width=0.92,label='MT')
p2 = ax1.bar(t,ve,width=0.92,bottom=mt,label='GP')
p3 = ax1.bar(t,bpp,width=0.92,bottom=mt+ve,color='red',label='BD')
p4 = ax1.bar(t,pv,width=0.92,bottom=mt+ve+bpp,label='PV+WT')
#p0 = plt.bar(t,bpm,width=0.92,color='brown',label='BC')
ax1.legend(loc='upper left')
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Power Output (kWh)')

ax2 = ax1.twinx()
plt.ylim((0,100))
ax2.plot(soc,linewidth=2,color='navy',marker='*',label='SOC',markersize=10)
ax2.set_ylabel('State of charge')
ax2.legend(loc='upper right')

############################# save result ####################################

foldername = 'DQN-S1 after S0'+' e='+str(epi)+' re='+str(int(re))+' cost='+str(int(cost))
if os.path.exists(foldername)==False:
    os.makedirs(foldername)
cwd = os.getcwd()
pdt.to_csv(cwd+'\\'+foldername+"\\pb.csv")
pd.DataFrame([re_max_l,re_mean_l]).to_csv(cwd+'\\'+foldername+"\\re_record.csv")
op_fig.savefig(cwd+'\\'+foldername+"\\op_fig.png",dpi=100)
re_fig.savefig(cwd+'\\'+foldername+"\\re_fig.png",dpi=100)

reop_fig = plt.gcf()
plt.show()
reop_fig.savefig(cwd+'\\'+foldername+"\\reop_fig.png",dpi=100)

############## Demonstration ##################
#5day DQN results
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
resultdata = pd.read_csv('pb.csv')
mt = np.array(pd.Series(resultdata['mt']))
gp = np.array(pd.Series(resultdata['ve']))
bp = np.array(pd.Series(resultdata['bp']))
re = np.array(pd.Series(resultdata['pv']))
ld = np.array(pd.Series(resultdata['ld']))
soc = np.array(pd.Series(resultdata['soc']))

bpp=[]
bpm=[]
zeros=np.zeros(120)
for i in range(120):
    if bp[i]>0:
        bpp.append(bp[i])
        bpm.append(0)
    else:
        bpm.append(bp[i])
        bpp.append(0)

t = np.arange(120)

fig3 = plt.figure(figsize=(15,6))
ax1 = fig3.add_subplot(111)
ax1.plot(ld-bpm,linewidth=2,color='black',marker='_',label='LD+EC+EH',markersize=10)
#ax1.plot(eh+ld-bpm,linewidth=2)
#ax1.plot(ld-bpm,linewidth=2)
ax1.plot(zeros-bpm,linewidth=2,color='brown',label='BC')
#ax1.plot(soc,linewidth=2,color='navy',marker='*',label='SOC',markersize=10)

#plt.bar(t,-ld,width=0.92,color='gray')
p1 = ax1.bar(t,mt,width=0.92,label='MT')
p2 = ax1.bar(t,gp,width=0.92,bottom=mt,label='GP')
p3 = ax1.bar(t,bpp,width=0.92,bottom=mt+gp,color='red',label='BD')
p4 = ax1.bar(t,re,width=0.92,bottom=mt+gp+bpp,label='PV+WT')
#p0 = plt.bar(t,bpm,width=0.92,color='brown',label='BC')
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Power Output (kWh)')

ax2 = ax1.twinx()
plt.ylim((0,100))
ax2.plot(soc,linewidth=2,color='navy',marker='*',label='SOC',markersize=10)
ax2.set_ylabel('State of charge (%)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

fig3.savefig("fig.png",dpi=100)
plt.show()

