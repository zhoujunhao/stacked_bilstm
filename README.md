# Stacked Bi-LSTM
This is the tensorflow implementation of Stacked Bi-LSTM using functional API for Chinese sentiment analysis. 

## Model Layout
<p align="center">
  <img width="550" height="250" src="https://github.com/zhoujunhao/stacked_bilstm/blob/master/figure/f2.PNG">
</p>

## Installation
- scikit-learn 0.19
- Tensorflow-gpu 1.5.0
- Python 2.7

## Train the model
**Run command below to train the model:**
- Train Stacked_BiLSTM model (in /4 Document representation + Sentiment Analysis). You can change different word embedding methods such as CBOW and Skip-gram.
```
python Stacked_BiLSTM.py
```

- Train baseline models. For example, you can train the machine learing model such as LR & SVM with SkipGram word embedding methods.
```
python Baseline_SkipGram_LR_SVM.py
```
Moreover, you can train different deep learning models as baseline models such as CNN and LSTM. Just change the parameters in `Stacked_BiLSTM.py` file.
```
# (4) Construct LSTM models
   #prediction,W=LSTM(lstmUnits,keepratio,data,numClasses)                    
   #prediction,W=BiLSTM(lstmUnits,keepratio,data,numClasses)                 
    prediction,W=stacked_BiLSTM(lstmUnits,keepratio,data,numClasses,num_layers)
   #prediction=CNN(lstmUnits,keepratio,data,kernel_size,numClasses)
   #prediction=stacked_CNN(lstmUnits,keepratio,data,kernel_size,numClasses)
```

## Visualization
**Run command below to visualize the impacts of differnet parameters:**
- Visualize the impacts of parameters including the number of LSTM cells and the maximum sentence length. For example, you can visualize the impacts of the number of LSTM cells for Stacked Bi-LSTM model. 
```
python Visualization_hidden_size.py.py
```

![Visualization](https://github.com/zhoujunhao/stacked_bilstm/blob/master/figure/f3.PNG)

## Citation
```
@article{zhou2019sentiment,
  title={Sentiment Analysis of Chinese Microblog Based on Stacked Bidirectional LSTM},
  author={Zhou, Junhao and Lu, Yue and Dai, Hong-Ning and Wang, Hao and Xiao, Hong},
  journal={IEEE Access},
  volume={7},
  pages={38856--38866},
  year={2019},
  publisher={IEEE}
}
```
