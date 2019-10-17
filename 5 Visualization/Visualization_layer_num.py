# -*- coding: utf-8 -*-
"""
Visualization layer num: 
    1. Load data from 'Data' folder
    2. Visualize loss and accuracy with different layer number

Created on Sat Apr 28 12:25:40 2018

@author: Yue, Lu
"""
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns



#读取数据
path=os.path.abspath('../Data')
#print('data')

#(0) 4 layers
name='StackedBLSTM_4layers_bu.txt'
file=io.open(path+'/'+name,'r')
line=file.readlines()

layer4_train_acc_set=[]
layer4_test_acc_set=[]
layer4_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    #print(line[i])
    train_test_loss=line[i].split('\t')
    #print(train_test_loss)
    #train_test_loss[i] = train_test_loss[i].replace('',)
    #print(train_test_loss)
    
    
    layer4_train_acc_set.append(train_test_loss[0].encode('utf-8'))
    layer4_test_acc_set.append(train_test_loss[1].encode('utf-8'))
    layer4_loss_set.append(train_test_loss[2].encode('utf-8'))
    
    layer4_train_acc_set_tmp = ','.join('%s' % id for id in layer4_train_acc_set)
    layer4_train_acc_set = map(float,layer4_train_acc_set)
    
    layer4_test_acc_set_tmp = ','.join('%s' % id for id in layer4_test_acc_set)
    layer4_test_acc_set = map(float,layer4_test_acc_set)
    
    layer4_loss_set_tmp = ','.join('%s' % id for id in layer4_loss_set)
    layer4_loss_set = map(float,layer4_loss_set)



#(1) 3 layers
#name='stackedBLSTM_3layer.txt'
name='StackedBLSTM_3layers.txt'
file=io.open(path+'/'+name,'r')
line=file.readlines()

layer3_train_acc_set=[]
layer3_test_acc_set=[]
layer3_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    #print(line[i])
    train_test_loss=line[i].split('\t')
    #print(train_test_loss)
    #train_test_loss[i] = train_test_loss[i].replace('',)
    #print(train_test_loss)
    
    
    layer3_train_acc_set.append(train_test_loss[0].encode('utf-8'))
    layer3_test_acc_set.append(train_test_loss[1].encode('utf-8'))
    layer3_loss_set.append(train_test_loss[2].encode('utf-8'))
    
    layer3_train_acc_set_tmp = ','.join('%s' % id for id in layer3_train_acc_set)
    layer3_train_acc_set = map(float,layer3_train_acc_set)
    
    layer3_test_acc_set_tmp = ','.join('%s' % id for id in layer3_test_acc_set)
    layer3_test_acc_set = map(float,layer3_test_acc_set)
    
    layer3_loss_set_tmp = ','.join('%s' % id for id in layer3_loss_set)
    layer3_loss_set = map(float,layer3_loss_set)
    
#    layer3_train_acc_set_tmp = ','.join(layer3_train_acc_set)
#    layer3_train_acc_set = map(float,layer3_train_acc_set)
#    
#    layer3_test_acc_set_tmp = ','.join(layer3_test_acc_set)
#    layer3_test_acc_set_set = map(float,layer3_test_acc_set)
#    
#    layer3_loss_set_tmp = ','.join(layer3_loss_set)
#    layer3_loss_set = map(float,layer3_loss_set)
    

#print(train_test_loss[0],train_test_loss[1],train_test_loss[2])

#(2) 2 layers
#name='stackedBLSTM.txt'
name='StackedBLSTM_2layers.txt'
file=io.open(path+'/'+name,'r')
line=file.readlines()

layer2_train_acc_set=[]
layer2_test_acc_set=[]
layer2_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #train_test_loss[i] = train_test_loss[i].replace('',)
    layer2_train_acc_set.append(train_test_loss[0].encode('utf-8') )
    layer2_test_acc_set.append(train_test_loss[1].encode('utf-8') )
    layer2_loss_set.append(train_test_loss[2].encode('utf-8') )
    
    #print(layer2_train_acc_set)
    layer2_train_acc_set_tmp = ','.join('%s' % id for id in layer2_train_acc_set)
    layer2_train_acc_set = map(float,layer2_train_acc_set)
    
    layer2_test_acc_set_tmp = ','.join('%s' % id for id in layer2_test_acc_set)
    layer2_test_acc_set = map(float,layer2_test_acc_set)
    
    layer2_loss_set_tmp = ','.join('%s' % id for id in layer2_loss_set)
    layer2_loss_set = map(float,layer2_loss_set)
    #print(layer2_train_acc_set)
#    
#    layer2_test_acc_set_tmp = ','.join(layer2_test_acc_set)
#    layer2_test_acc_set_set = map(float,layer2_test_acc_set)
#    
#    layer2_loss_set_tmp = ','.join(layer2_loss_set)
#    layer2_loss_set = map(float,layer2_loss_set)

    #print(layer2_train_acc_set)
#print('layer2_loss_set:',layer2_loss_set)
##print('layer3_loss_set:',layer3_loss_set)
#layer2_loss_set_tmp = ','.join(layer2_loss_set)
#layer2_loss_set = map(float,layer2_loss_set_tmp)
#
#layer3_loss_set_tmp = ','.join(layer3_loss_set)
#layer3_loss_set = map(float,layer3_loss_set_tmp)


#模型可视化
#(1) Loss
sns.set_style('whitegrid')

sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))
plt.plot(layer2_loss_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer3_loss_set,sns.xkcd_rgb['blue'],label='3 layers')
plt.plot(layer4_loss_set,sns.xkcd_rgb['red'],label='4 layers')


plt.xlabel('Number of Epochs')
plt.ylabel('Average Loss')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.tight_layout()
plt.legend(loc='upper right', fontsize=12,frameon=True,shadow=True)
plt.show()

###############################
#loss layer2 green
sns.set_style('whitegrid')

#sns.boxplot(x = layer2_loss_set['class'],y = layer2_loss_set['sepal width']

#df_iris = pd.read_csv('../input/iris.csv')
#sns.boxplot(x = df_iris['class'],y = df_iris['sepal width'])
#plt.show()
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))
plt.plot(layer2_loss_set,sns.xkcd_rgb['green'],label='2 layers')
#plt.plot(layer3_loss_set,sns.xkcd_rgb['blue'],label='3 layers')
#sns.boxplot(x=,y=layer2_loss_set)
#plt.show()


plt.xlabel('Number of Epochs')
plt.ylabel('Average Loss')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.tight_layout()
plt.legend(loc='upper right', fontsize=12,frameon=True,shadow=True)
plt.show()

#loss layer3 blue
sns.set_style('whitegrid')

sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))
#plt.plot(layer2_loss_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer3_loss_set,sns.xkcd_rgb['blue'],label='3 layers')

plt.xlabel('Number of Epochs')
plt.ylabel('Average Loss')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.tight_layout()
plt.legend(loc='upper right', fontsize=12,frameon=True,shadow=True)
plt.show()

#loss layer4 red
sns.set_style('whitegrid')

sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))
#plt.plot(layer2_loss_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer4_loss_set,sns.xkcd_rgb['red'],label='4 layers')

plt.xlabel('Number of Epochs')
plt.ylabel('Average Loss')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.tight_layout()
plt.legend(loc='upper right', fontsize=12,frameon=True,shadow=True)
plt.show()

##############################
#(2) Training Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(layer2_train_acc_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer3_train_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
plt.plot(layer4_train_acc_set,sns.xkcd_rgb['red'],label='4 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.tight_layout()
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()

#################Training accuracy 2layers
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(layer2_train_acc_set,sns.xkcd_rgb['green'],label='2 layers')
#plt.plot(layer3_train_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
#plt.plot(layer4_train_acc_set,sns.xkcd_rgb['red'],label='4 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.tight_layout()
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()

#################Training accuracy 3layers
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


#plt.plot(layer2_train_acc_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer3_train_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
#plt.plot(layer4_train_acc_set,sns.xkcd_rgb['red'],label='4 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.tight_layout()
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()

#################Training accuracy 4layers
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


#plt.plot(layer2_train_acc_set,sns.xkcd_rgb['green'],label='2 layers')
#plt.plot(layer3_train_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
plt.plot(layer4_train_acc_set,sns.xkcd_rgb['red'],label='4 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.tight_layout()
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()


#(3) Validation Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(layer2_test_acc_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer3_test_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
plt.plot(layer4_test_acc_set,sns.xkcd_rgb['red'],label='4 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()
#plt.savefig("./eps/layer_num/loss.eps", format='eps', dpi=1000)

#(3) Validation Accuracy 2layers
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(layer2_test_acc_set,sns.xkcd_rgb['green'],label='2 layers')
#plt.plot(layer3_test_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
#plt.plot(layer4_test_acc_set,sns.xkcd_rgb['red'],label='4 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()


#(3) Validation Accuracy 3layers
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


#plt.plot(layer2_test_acc_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer3_test_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
#plt.plot(layer4_test_acc_set,sns.xkcd_rgb['red'],label='4 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()

#(3) Validation Accuracy 4layers
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


#plt.plot(layer2_test_acc_set,sns.xkcd_rgb['green'],label='2 layers')
#plt.plot(layer3_test_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
plt.plot(layer4_test_acc_set,sns.xkcd_rgb['red'],label='4 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()



