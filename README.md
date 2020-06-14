Deep learning model for sentiment analysis on IMDB movie reviews dataset provided on Kaggle.

# results
```
Epoch 3/3
 - 743s - loss: 0.4784 - acc: 0.9376 - val_loss: 0.5658 - val_acc: 0.9026
```
# model summary
```
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 1449, 8)           1081296
_________________________________________________________________
lstm_1 (LSTM)                (None, 1449, 10)          760
_________________________________________________________________
flatten_1 (Flatten)          (None, 14490)             0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 14491
=================================================================
Total params: 1,096,547
Trainable params: 1,096,547
Non-trainable params: 0
_____________________________
```