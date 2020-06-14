from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Flatten
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.constraints import max_norm
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import pickle
import re

EMBEDDING_DIM = 8

df = pd.read_csv('IMDB dataset.csv')
#df2 = pd.read_csv('rt-polarity.pos', sep='\n', header=None)
df.replace({'sentiment': {'positive': 1, 'negative': 0}}, inplace = True)

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

def preprocess_reviews(reviews):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

reviews_clean = preprocess_reviews(list(df['review']))
reviews_clean = remove_stop_words(reviews_clean)
df['review'] = reviews_clean
print(df)

label = df['sentiment'].values
data = df['review'].values

X_train, X_test, y_train, y_test = train_test_split(data , label, test_size = 0.2, random_state=1)

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(data)
x = []
for s in data:
    y = re.split(",| , | |, ", s)
    x.append(len(y))
#max_length = max([len(re.split(',| | , | ,|, ', s)) for s in data])
max_length = max(x)
print(max_length)
vocab_size = len(tokenizer_obj.word_index) + 1

X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)

X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_tokens, maxlen = max_length, padding= 'post')
X_test_pad = pad_sequences(X_test_tokens, maxlen = max_length, padding= 'post')

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length = max_length))
model.add(LSTM(units = 10, dropout = 0.5, return_sequences = True))
#model.add(LSTM(units = 60, dropout= 0.2, recurrent_dropout = 0.2, return_sequences = True))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01)))

model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics = ['accuracy'])
model.summary()

model.fit(X_train_pad, y_train, batch_size= 64 ,epochs= 3, validation_data=(X_test_pad, y_test), verbose = 2)

pickle.dump(model, open('model.pkl','wb'))
