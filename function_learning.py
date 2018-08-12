import numpy as np
import keras
import  keras.backend as k
from keras.models import *
from keras.layers import *
from keras.initializers import *

from  nalu import NALU

x_shape = [10000,100]

#x=np.random.rand(x_shape)*10
#test_ex=np.random.rand(x_shape)*100

arithmetic_functions={
'add': lambda x,y :x+y,
'sub': lambda x,y:x-y,
'mul': lambda x,y: x*y,
'div': lambda x,y: x/y,
'square': lambda x,y: np.square(x),
'sqrt' : lambda x,y : np.sqrt(x),
}

def nalu(mode=NALU):
    x = Input((100,))
    y = NALU(2, mode=mode, 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(x)
    y = NALU(1, mode=mode, 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(y)
    return Model(x, y)
    
def mlp():
    x = Input((100,))
    y = Dense(2, activation="relu")(x)
    y = Dense(1)(y)
    return Model(x, y)

def get_data(N, op):
    split = 45
    X_train = np.random.normal(0, 0.5, (N, 100))
    a = X_train[:, :split].sum(1)
    b = X_train[:, split:].sum(1)
    print(a.min(), a.max(), b.min(), b.max())
    Y_train = op(a, b)[:, None]
    X_val = np.random.normal(0, 2, (N, 100))
    a = X_val[:, :split].sum(1)
    b = X_val[:, split:].sum(1)
    print(a.min(), a.max(), b.min(), b.max())
    Y_val = op(a, b)[:, None]
	return (X_train,Y_train),(X_val,Y_val)
	
if __name__=='__main__':
	#nalu
	for op in arithmetic_functions:
		print (op)
		m=nalu('NALU')
		m.compile(loss='adam',metrics=['mse'])
		(X_train,Y_train),(X_val,Y_val) = get_data(2 ** 16,arithmetic_functions[op])
		k.set_value(m.optimizer.lr, 1e-2)
		m.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1024, epochs=200)
    	k.set_value(m.optimizer.lr, 1e-3)
		m.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1024, epochs=200)
		k.set_value(m.optimizer.lr, 1e-4)
		m.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1024, epochs=200)
