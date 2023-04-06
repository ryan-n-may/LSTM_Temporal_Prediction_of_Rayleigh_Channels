# LSTM-time-series-prediction-of-a-Rayleigh-Channel
Using an LSTM to train LS estimations of a Rayleigh channel, and outputting a channel forecast. 
64 symbols are QAM modulated and transmitted across a Rayleigh Channel.  LS and MMSE estimation is performed using pilot signals. 
An LSTM neural network is trained on 64 QAM symbols and then forecasts the Rayleigh channel fading properties.  

### Visualising QAM modulated symbols after equalisation
![QAM modulation scatterplots](https://github.com/Meandi-n/LSTM-time-series-prediction-of-a-Rayleigh-Channel/blob/main/example1figures/modulated_symbols.png)

The LSTM network equaliser has less performance than LS and MMSE estimation techniques.  
This is under low doppler shift conditions and less than 5 Rayleigh channel paths.
The LSTM network used contains one LSTM layer of 250 hidden units. 
A well trained LSTM should outperform classical LS and MMSE estimations. A performance lack is an indication
of an implementation issue. 

### Training data for LSTM network
![QAM modulation scatterplots](https://github.com/Meandi-n/LSTM-time-series-prediction-of-a-Rayleigh-Channel/blob/main/example1figures/LS_estimations.png)

The channel data used to train the LSTM network for estimation are LS estimations performed over 64 pilot sequences.  Variation is due to noise and multipath
fading. 

### Accuracy of LSTM forecasting
![QAM modulation scatterplots](https://github.com/Meandi-n/LSTM-time-series-prediction-of-a-Rayleigh-Channel/blob/main/example1figures/performance.png)

A possible source of performance loss is observed here.  The LSTM forecasting lags behind the actual Rayleigh channel transfer function. 

