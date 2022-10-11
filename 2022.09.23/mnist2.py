from matplotlib import pyplot as plt
import tensorflow as tf
#from tensorflow import keras
#Используем костыль для исправления IntelliSense для keras по гайду:
#https://stackoverflow.com/questions/71000250/import-tensorflow-keras-could-not-be-resolved-after-upgrading-to-tensorflow-2
import keras.api._v2.keras as keras
from keras import layers
from keras import losses
from keras.datasets import mnist
import numpy as np


(x_train,y_train), (x_test1,y_test)=mnist.load_data()
print("Train X=%s, y=%s"%(x_train.shape,y_train.shape))
print("Test X=%s, y=%s"%(x_test1.shape,y_test.shape))

x_train=x_train.reshape(-1,28*28).astype('float32')/255.0
x_test=x_test1.reshape(-1,28*28).astype('float32')/255.0
print("Train X=%s, y=%s"%(x_train.shape,y_train.shape))
print("Test X=%s, y=%s"%(x_test.shape,y_test.shape))
print(x_test[0].shape)
model=keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        #layers.Dense(8192,activation='relu6'),
        #layers.Dense(4096,activation='relu6'),
        #layers.Dense(2048,activation='relu'),
        #layers.Dense(1024,activation='relu'),
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
model.fit(x_train,y_train,batch_size=32,epochs=3,verbose=2)
results=model.evaluate(x_test,y_test,batch_size=32,verbose=2)
print(str(results))
value=np.random.randint(0,10000)

print(x_test[value].shape)

single=x_test[value]
image=np.zeros((28,28,3))
print(image.shape)

for y in range(0,image.shape[0]):
    for x in range(0,image.shape[1]):
        for c in range(0,image.shape[2]):
            image[y,x,c]=single[y*28+x]



print(single.shape)
#print(single)

singleReady=np.zeros((1,28*28))

for y in range(0,image.shape[0]):
    for x in range(0,image.shape[1]):
            singleReady[0][y*28+x]=single[y*28+x]

print(model.predict(singleReady,batch_size=1).argmax())

plt.imshow(image)
plt.show()