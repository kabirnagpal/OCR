import pandas as pd
import numpy as np

dataset=pd.read_csv('/home/jak/Desktop/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv')

y=dataset.iloc[:,0].values
x=dataset.iloc[:,1:].values

x1=[]
for i in range (0,x.shape[0]):
    B = np.reshape(x[i,:], (-1, 28))
    x1.append(np.asarray(B,dtype=np.uint8))
    
x1 = np.asarray(x1,dtype=np.uint8)
x1 = x1.reshape(len(x1),28,28,1)

from keras.utils import np_utils
y = np_utils.to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y, test_size=0.2,random_state=0)

from keras.models import Sequential
from keras.layers import Convolution2D    as Conv2D #for dealing with pictures
from keras.layers import Flatten   , MaxPooling2D      #flatten the maps
from keras.layers import Dense,Dropout

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train,validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)

import pickle
with open('/home/jak/Desktop/writing99', 'wb') as f:
    pickle.dump(model,f)
    
y_pred = model.predict(x_test)

for i in range(0,y_pred.shape[0]):
    maxi = max(y_pred[i])
    for j in range(0,y_pred.shape[1]):
        if y_pred[i][j]==maxi:
            y_pred[i][j]=1
        else:
            y_pred[i][j]=0

y_pred=y_pred.astype(int)
Ypred=[]
for i in range(0,y_pred.shape[0]):
    Ypred.append(np.argmax(y_pred[i]))

y_pred = np.asarray(Ypred)

Ytest=[]
for i in range(0,y_test.shape[0]):
    Ytest.append(np.argmax(y_test[i]))

Y_test = np.asarray(Ytest)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

n=cm.shape[0]
sum_second_diagonal=sum([cm[i][j] for i in range(n) for j in range(n) if i==j])

print(sum_second_diagonal*100/x_test.shape[0])