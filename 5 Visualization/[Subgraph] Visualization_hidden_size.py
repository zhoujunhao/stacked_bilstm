# -*- coding: utf-8 -*-
"""
[Subgraph] Visualization hidden size: 
    1. Load data from 'Data' folder
    2. Visualize loss and accuracy with different hidden size

Created on Sat Apr 28 12:25:40 2018

@author: Yue, Lu
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns


#读取数据
path=os.path.abspath('../Data')

#(1)32 LSTM cells
name='stackedBLSTM_32cell.txt'
file=open(path+'\\'+name,'r')
line=file.readlines()

cell32_train_acc_set=[]
cell32_test_acc_set=[]
cell32_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    cell32_train_acc_set.append(train_test_loss[0])
    cell32_test_acc_set.append(train_test_loss[1])
    cell32_loss_set.append(train_test_loss[2])
    

#(2)64 LSTM cells
name='stackedBLSTM.txt'
file=open(path+'\\'+name,'r')
line=file.readlines()

cell64_train_acc_set=[]
cell64_test_acc_set=[]
cell64_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    cell64_train_acc_set.append(train_test_loss[0])
    cell64_test_acc_set.append(train_test_loss[1])
    cell64_loss_set.append(train_test_loss[2])
    
   
#(3)128 LSTM cells
name='stackedBLSTM_128cell.txt'
file=open(path+'\\'+name,'r')
line=file.readlines()

cell128_train_acc_set=[]
cell128_test_acc_set=[]
cell128_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    cell128_train_acc_set.append(train_test_loss[0])
    cell128_test_acc_set.append(train_test_loss[1])
    cell128_loss_set.append(train_test_loss[2])
    


#模型可视化
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":1.5})

fig=plt.figure(figsize=(15,3))
plt1=fig.add_subplot(1,3,1)
plt2=fig.add_subplot(1,3,2)
plt3=fig.add_subplot(1,3,3)

#(1) Loss
#subplot1
plt1.plot(cell32_loss_set,sns.xkcd_rgb['orange'],label='32 LSTM cells')
plt1.plot(cell64_loss_set,sns.xkcd_rgb['green'],label='64 LSTM cells')
plt1.plot(cell128_loss_set,sns.xkcd_rgb['blue'],label='128 LSTM cells')
plt1.set_xlabel('Number of Epochs')
plt1.set_ylabel('Average Loss')



#(2) Training Accuracy
#subplot2
plt2.plot(cell32_train_acc_set,sns.xkcd_rgb['orange'],label='32 LSTM cells')
plt2.plot(cell64_train_acc_set,sns.xkcd_rgb['green'],label='64 LSTM cells')
plt2.plot(cell128_train_acc_set,sns.xkcd_rgb['blue'],label='128 LSTM cells')
plt2.set_xlabel('Number of Epochs')
plt2.set_ylabel('Training Accuracy')


#(3) Validation Accuracy
#subplot3
plt3.plot(cell32_test_acc_set,sns.xkcd_rgb['orange'],label='32 LSTM cells')
plt3.plot(cell64_test_acc_set,sns.xkcd_rgb['green'],label='64 LSTM cells')
plt3.plot(cell128_test_acc_set,sns.xkcd_rgb['blue'],label='128 LSTM cells')
plt3.set_xlabel('Number of Epochs')
plt3.set_ylabel('Validation Accuracy')




#图例设置
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()