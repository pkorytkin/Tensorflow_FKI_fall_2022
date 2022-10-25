import tensorflow as tf
#from tensorflow import keras
#Используем костыль для исправления IntelliSense для keras по гайду:
#https://stackoverflow.com/questions/71000250/import-tensorflow-keras-could-not-be-resolved-after-upgrading-to-tensorflow-2
import keras.api._v2.keras as keras
from keras import layers
from keras import losses
from keras.datasets import mnist
import numpy


(x_train,y_train), (x_test,y_test)=mnist.load_data()
print("Train X=%s, y=%s"%(x_train.shape,y_train.shape))
print("Test X=%s, y=%s"%(x_test.shape,y_test.shape))

x_train=x_train.reshape(-1,28*28).astype('float32')/255.0
x_test=x_test.reshape(-1,28*28).astype('float32')/255.0
print("Train X=%s, y=%s"%(x_train.shape,y_train.shape))
print("Test X=%s, y=%s"%(x_test.shape,y_test.shape))

model=keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512,activation='relu'),
        layers.Dense(256,activation='relu'),
        layers.Dense(10)
    ]
 )
model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)