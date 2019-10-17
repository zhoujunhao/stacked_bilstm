# -*- coding: utf-8 -*-
"""
Visualization hidden size: 
    1. Load data from 'Data' folder
    2. Visualize loss and accuracy with different hidden size

Created on Sat Apr 28 12:25:40 2018

@author: Yue, Lu
"""
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns


#读取数据
path=os.path.abspath('../Data')

#(1)32 LSTM cells
name='stackedBLSTM_32cell.txt'
file=io.open(path+'/'+name,'r')
line=file.readlines()

cell32_train_acc_set=[]
cell32_test_acc_set=[]
cell32_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    cell32_train_acc_set.append(train_test_loss[0].encode('utf-8'))
    cell32_test_acc_set.append(train_test_loss[1].encode('utf-8'))
    cell32_loss_set.append(train_test_loss[2].encode('utf-8'))
    
    cell32_train_acc_set_tmp = ','.join('%s' % id for id in cell32_train_acc_set)
    cell32_train_acc_set = map(float,cell32_train_acc_set)
    
    cell32_test_acc_set_tmp = ','.join('%s' % id for id in cell32_test_acc_set)
    cell32_test_acc_set = map(float,cell32_test_acc_set)
    
    cell32_loss_set_tmp = ','.join('%s' % id for id in cell32_loss_set)
    cell32_loss_set = map(float,cell32_loss_set)
    

#(2)64 LSTM cells
name='stackedBLSTM.txt'
file=io.open(path+'/'+name,'r')
line=file.readlines()

cell64_train_acc_set=[]
cell64_test_acc_set=[]
cell64_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    cell64_train_acc_set.append(train_test_loss[0].encode('utf-8'))
    cell64_test_acc_set.append(train_test_loss[1].encode('utf-8'))
    cell64_loss_set.append(train_test_loss[2].encode('utf-8'))
    
    cell64_train_acc_set_tmp = ','.join('%s' % id for id in cell64_train_acc_set)
    cell64_train_acc_set = map(float,cell64_train_acc_set)
    
    cell64_test_acc_set_tmp = ','.join('%s' % id for id in cell64_test_acc_set)
    cell64_test_acc_set = map(float,cell64_test_acc_set)
    
    cell64_loss_set_tmp = ','.join('%s' % id for id in cell64_loss_set)
    cell64_loss_set = map(float,cell64_loss_set)
    
   
#(3)128 LSTM cells
name='stackedBLSTM_128cell.txt'
file=io.open(path+'/'+name,'r')
line=file.readlines()

cell128_train_acc_set=[]
cell128_test_acc_set=[]
cell128_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    cell128_train_acc_set.append(train_test_loss[0].encode('utf-8'))
    cell128_test_acc_set.append(train_test_loss[1].encode('utf-8'))
    cell128_loss_set.append(train_test_loss[2].encode('utf-8'))
    
    cell128_train_acc_set_tmp = ','.join('%s' % id for id in cell128_train_acc_set)
    cell128_train_acc_set = map(float,cell128_train_acc_set)
    
    cell128_test_acc_set_tmp = ','.join('%s' % id for id in cell128_test_acc_set)
    cell128_test_acc_set = map(float,cell128_test_acc_set)
    
    cell128_loss_set_tmp = ','.join('%s' % id for id in cell128_loss_set)
    cell128_loss_set = map(float,cell128_loss_set)
    



#模型可视化
#(1) Loss
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(cell32_loss_set,sns.xkcd_rgb['orange'],label='32 LSTM cells')
plt.plot(cell64_loss_set,sns.xkcd_rgb['green'],label='64 LSTM cells')
plt.plot(cell128_loss_set,sns.xkcd_rgb['blue'],label='128 LSTM cells')
plt.xlabel('Number of Epochs')
plt.ylabel('Average Loss')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='upper right', fontsize=12,frameon=True,shadow=True)
#plt.show()
plt.savefig("./eps/hidden_num/loss.eps", format='eps', dpi=1000)

#(2) Training Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(cell32_train_acc_set,sns.xkcd_rgb['orange'],label='32 LSTM cells')
plt.plot(cell64_train_acc_set,sns.xkcd_rgb['green'],label='64 LSTM cells')
plt.plot(cell128_train_acc_set,sns.xkcd_rgb['blue'],label='128 LSTM cells')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
#plt.show()
plt.savefig("./eps/hidden_num/training_acc.eps", format='eps', dpi=1000)

#(3) Validation Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(cell32_test_acc_set,sns.xkcd_rgb['orange'],label='32 LSTM cells')
plt.plot(cell64_test_acc_set,sns.xkcd_rgb['green'],label='64 LSTM cells')
plt.plot(cell128_test_acc_set,sns.xkcd_rgb['blue'],label='128 LSTM cells')
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
#plt.show()
plt.savefig("./eps/hidden_num/validation_acc.eps", format='eps', dpi=1000)


