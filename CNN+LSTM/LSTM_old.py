import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
import csv
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input
from keras.layers import Dense, LSTM, Dropout, Embedding, Input, Activation, Bidirectional, TimeDistributed, RepeatVector, Flatten
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential, Model

learning_rate=0.001
look_back=20
batch_size=5
hidden_nodes = 256
epochs =  100

adam = Adam(lr=learning_rate)


def create_dataset_input(dataset, look_back):
	dataX = []
	for i in range(len(dataset)-look_back):
		dataX.append(dataset[i:(i+look_back)])
	return np.array(dataX)

def mode_decide(input_mode):
    train_mode=input_mode.split('-',1)[0]
    val_mode=input_mode.split('-',1)[1]
    error1=(train_mode!='a')and(train_mode!='b')and(train_mode!='ab')
    error2=(train_mode!='a')and(train_mode!='b')and(train_mode!='ab')
    if error1 or error2:
        raise ValueError # Wrong input mode type
    mode={'train_set':train_mode,'val_set':val_mode}
    return mode

def load_data(mode):
    filename=[mode['train_set']+'_train_set'+'.csv']
    filename="".join(filename)
    train_data=pd.read_csv(filename)

    filename=[mode['val_set']+'_val_set'+'.csv']
    filename="".join(filename)
    val_data=pd.read_csv(filename)
    return train_data,val_data

def data_prepocess(train_data,val_data,batch_size=batch_size,look_back=look_back):
	#train_set 设置
	train_raw_x=train_data['Loc_x']
	train_raw_x=np.array(train_raw_x).astype(float).reshape(-1,1)
	scaler_loc_x=MinMaxScaler()
	train_loc_x=scaler_loc_x.fit_transform(train_raw_x)

	train_raw_y=train_data['Loc_y']
	train_raw_y=np.array(train_raw_y).astype(float).reshape(-1,1)
	scaler_loc_y=MinMaxScaler()
	train_loc_y=scaler_loc_y.fit_transform(train_raw_y)

	train_Mag_x=train_data['GeoX']
	train_Mag_x=np.array(train_Mag_x).astype(float).reshape(-1,1)
	scaler_mag_x=MinMaxScaler()
	Mag_x=scaler_mag_x.fit_transform(train_Mag_x)

	train_Mag_y=train_data['GeoY']
	train_Mag_y=np.array(train_Mag_y).astype(float).reshape(-1,1)
	scaler_mag_y=MinMaxScaler()
	Mag_y=scaler_mag_y.fit_transform(train_Mag_y)

	train_Mag_z=train_data['GeoZ']
	train_Mag_z=np.array(train_Mag_z).astype(float).reshape(-1,1)
	scaler_mag_z=MinMaxScaler()
	Mag_z=scaler_mag_z.fit_transform(train_Mag_z)


	train_size=int(len(train_loc_x))
	#val_set 设置
	
	val_raw_x=val_data['Loc_x']
	val_raw_x=np.array(val_raw_x).astype(float).reshape(-1,1)
	v_scaler_loc_x=MinMaxScaler()
	val_loc_x=v_scaler_loc_x.fit_transform(val_raw_x)

	val_raw_y=val_data['Loc_y']
	val_raw_y=np.array(val_raw_y).astype(float).reshape(-1,1)
	v_scaler_loc_y=MinMaxScaler()
	val_loc_y=v_scaler_loc_y.fit_transform(val_raw_y)

	val_Mag_x=val_data['GeoX']
	val_Mag_x=np.array(val_Mag_x).astype(float).reshape(-1,1)
	v_scaler_mag_x=MinMaxScaler()
	val_Mag_x=v_scaler_mag_x.fit_transform(val_Mag_x)

	val_Mag_y=val_data['GeoY']
	val_Mag_y=np.array(val_Mag_y).astype(float).reshape(-1,1)
	v_scaler_mag_y=MinMaxScaler()
	val_Mag_y=v_scaler_mag_y.fit_transform(val_Mag_y)

	val_Mag_z=val_data['GeoZ']
	val_Mag_z=np.array(val_Mag_z).astype(float).reshape(-1,1)
	v_scaler_mag_z=MinMaxScaler()
	val_Mag_z=v_scaler_mag_z.fit_transform(val_Mag_z)

	val_size=int(len(val_loc_x))
	
	
	train_mag_x = create_dataset_input(train_Mag_x, look_back = look_back)
	train_mag_y = create_dataset_input(train_Mag_y, look_back = look_back)
	train_mag_z = create_dataset_input(train_Mag_z, look_back = look_back)

	
	test_mag_x = create_dataset_input(val_Mag_x, look_back = look_back)
	test_mag_y = create_dataset_input(val_Mag_y, look_back = look_back)
	test_mag_z = create_dataset_input(val_Mag_z, look_back = look_back)
	
	#print('trian_mag_x:',train_mag_x)
	
	train_loc_x = create_dataset_input(train_loc_x, look_back = look_back)
	train_loc_y = create_dataset_input(train_loc_y, look_back = look_back)
	test_loc_x = create_dataset_input(val_loc_x, look_back = look_back)
	test_loc_y = create_dataset_input(val_loc_y, look_back = look_back)


	trainX = np.concatenate((train_mag_x,train_mag_y,train_mag_z),axis = 2)
	testX = np.concatenate((test_mag_x,test_mag_y,test_mag_z),axis = 2)
	#print('train_loc_x.shape:',train_loc_x.shape)
	trainY = np.concatenate((train_loc_x,train_loc_y),axis = 2)
	testY = np.concatenate((test_loc_x,test_loc_y),axis = 2)
	trainY = np.reshape(trainY, (len(trainY),look_back,2))
	#print('trianY:',trainY.shape)
	lengthTrain = len(trainX)
	lengthTest = len(testX)
	while(lengthTrain % batch_size != 0):
		lengthTrain -= 1
	while(lengthTest % batch_size != 0):
		lengthTest -= 1

	return trainX[0:lengthTrain],trainY[0:lengthTrain],testX[0:lengthTest],testY[0:lengthTest]

def model_train(train_x, train_y, test_x, test_y,file_structure,file_acc2loss):
	model=model_build()
	for i in range(epochs):
		history = model.fit(train_x, train_y, batch_size=batch_size, epochs = 1, verbose=1,shuffle = False) #validation_split=0.1, validation_data=(test_x, test_y)
	#	# need to reset state for every epoch
		model.reset_states()
	#	#print('hidden_state:',hidden_state)
	#	# list all data in history
	#	'''
	#	print('history.keys()',hist.history.keys())
	#	# summarize history for accuracy
	#	plt.plot(hist.history['acc'])
	#	plt.plot(hist.history['val_acc'])
	#	plt.title('model accuracy')
	#	plt.ylabel('accuracy')
	#	plt.xlabel('epoch')
	#	plt.legend(['train', 'test'], loc='upper left')
	#	plt.show()
	#	'''
		print('Real Epoches:',i+1)
		with open(file_acc2loss,'a', newline='') as csvfile:
			if not os.path.getsize(file_acc2loss):        #file is empty
				spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
				spamwriter.writerow(['epochs','loss','acc'])#, 'val_loss','val_acc' 
				
			data = ([
				i,history.history['loss'][0],history.history['acc'][0]#, history.history['val_loss'][0], history.history['val_acc'][0]
				])  
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow(data)
	return model

def model_build(hidden_nodes=hidden_nodes,batch_size=batch_size , time_steps = look_back, feature_size = 3):
	inputs1 = Input(batch_shape = (batch_size,look_back,feature_size))
	lstm1 = LSTM(hidden_nodes, stateful = True, return_sequences=True, return_state=True,dropout=0.2)(inputs1)
	lstm1 = LSTM(hidden_nodes,return_sequences=True,dropout=0.2)(lstm1)
	lstm1 = TimeDistributed(Dense((2)))(lstm1)
	model = Model(input = inputs1, outputs = lstm1)
	print(model.layers)
	model.compile(loss='mean_squared_error', optimizer=adam,metrics=['acc'])
	model.summary()
	return model

if __name__=='__main__':
	input_mode=[]
	change=False
	if change:
		input_mode=input('Please inport train and val mode in _-_(e.g:a-b)\n')
		if input_mode=='all' :
			input_mode=['a-a','a-b','a-ab','b-a','b-b','b-ab','ab-a','ab-b','ab-ab']
	else:
		input_mode=['a-b']
	
	for t_v in input_mode:
		mode=mode_decide(t_v)
		file_structure = [mode['train_set']+'-'+mode['val_set']+'_'+'model_ts=30_256_5_100.png']
		file_acc2loss = [mode['train_set']+'-'+mode['val_set']+'_'+'log_ts=30_256_5_100.csv']
		file_structure="".join(file_structure)
		file_acc2loss = "".join(file_acc2loss)
		train_data,val_data=load_data(mode)
		train_x, train_y, test_x, test_y=data_prepocess(train_data,val_data)
		model=model_train(train_x, train_y, test_x, test_y,file_structure,file_acc2loss)
		del model
		