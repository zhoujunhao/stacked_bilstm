# -*- coding: utf-8 -*-
"""
Visualization input size: 
    1. Load data from 'Data' folder
    2. Visualize loss and accuracy with different input size

Created on Sat Apr 28 12:25:40 2018

@author: Yue, Lu
"""
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns


#读取数据
path=os.path.abspath('../Data')

#(1)7 words
name='stackedBLSTM_7words.txt'
file=io.open(path+'/'+name,'r')
line=file.readlines()

word7_train_acc_set=[]
word7_test_acc_set=[]
word7_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    word7_train_acc_set.append(train_test_loss[0].encode('utf-8'))
    word7_test_acc_set.append(train_test_loss[1].encode('utf-8'))
    word7_loss_set.append(train_test_loss[2].encode('utf-8'))
    
    word7_train_acc_set_tmp = ','.join('%s' % id for id in word7_train_acc_set)
    word7_train_acc_set = map(float,word7_train_acc_set)
    
    word7_test_acc_set_tmp = ','.join('%s' % id for id in word7_test_acc_set)
    word7_test_acc_set = map(float,word7_test_acc_set)
    
    word7_loss_set_tmp = ','.join('%s' % id for id in word7_loss_set)
    word7_loss_set = map(float,word7_loss_set)
    

#(2)13 words
name='stackedBLSTM.txt'
file=io.open(path+'/'+name,'r')
line=file.readlines()

word13_train_acc_set=[]
word13_test_acc_set=[]
word13_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    word13_train_acc_set.append(train_test_loss[0].encode('utf-8'))
    word13_test_acc_set.append(train_test_loss[1].encode('utf-8'))
    word13_loss_set.append(train_test_loss[2].encode('utf-8'))
    
    word13_train_acc_set_tmp = ','.join('%s' % id for id in word13_train_acc_set)
    word13_train_acc_set = map(float,word13_train_acc_set)
    
    word13_test_acc_set_tmp = ','.join('%s' % id for id in word13_test_acc_set)
    word13_test_acc_set = map(float,word13_test_acc_set)
    
    word13_loss_set_tmp = ','.join('%s' % id for id in word13_loss_set)
    word13_loss_set = map(float,word13_loss_set)

    
    
#(3) 20 words
name='stackedBLSTM_10words.txt'
file=io.open(path+'/'+name,'r')
line=file.readlines()

word10_train_acc_set=[]
word10_test_acc_set=[]
word10_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    word10_train_acc_set.append(train_test_loss[0].encode('utf-8'))
    word10_test_acc_set.append(train_test_loss[1].encode('utf-8'))
    word10_loss_set.append(train_test_loss[2].encode('utf-8'))
    
    word10_train_acc_set_tmp = ','.join('%s' % id for id in word10_train_acc_set)
    word10_train_acc_set = map(float,word10_train_acc_set)
    
    word10_test_acc_set_tmp = ','.join('%s' % id for id in word10_test_acc_set)
    word10_test_acc_set = map(float,word10_test_acc_set)
    
    word10_loss_set_tmp = ','.join('%s' % id for id in word10_loss_set)
    word10_loss_set = map(float,word10_loss_set)



#模型可视化
#(1) Loss
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))

plt.plot(word7_loss_set,sns.xkcd_rgb['orange'],label='7 words')
plt.plot(word10_loss_set,sns.xkcd_rgb['green'],label='10 words')
plt.plot(word13_loss_set,sns.xkcd_rgb['blue'],label='13 words')

plt.xlabel('Number of Epochs')
plt.ylabel('Average Loss')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='upper right', fontsize=12,frameon=True,shadow=True)
#plt.show()
plt.savefig("./eps/loss.eps", format='eps', dpi=1000)

#(2) Training Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(word7_train_acc_set,sns.xkcd_rgb['orange'],label='7 words')
plt.plot(word10_train_acc_set,sns.xkcd_rgb['green'],label='10 words')
plt.plot(word13_train_acc_set,sns.xkcd_rgb['blue'],label='13 words')

plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
#plt.show()
plt.savefig("./eps/training_acc.eps", format='eps', dpi=1000)

#(3) Validation Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(word7_test_acc_set,sns.xkcd_rgb['orange'],label='7 words')
plt.plot(word10_test_acc_set,sns.xkcd_rgb['green'],label='10 words')
plt.plot(word13_test_acc_set,sns.xkcd_rgb['blue'],label='13 words')

plt.xlabel('Number of Epochs')
plt.ylabel('Validation Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
#plt.show()
plt.savefig("./eps/validation_acc.eps", format='eps', dpi=1000)


