"""
This file is the combination of RSS data and Geomagnetic field data

"""




from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import argparse
import numpy as np
import tensorflow as tf
import time
import os
import csv
from sklearn.preprocessing import MinMaxScaler
from keras.utils import plot_model
from keras.layers import Input
from keras.layers import Dense, LSTM, Dropout, Embedding, Input, Activation, Bidirectional, TimeDistributed, RepeatVector,Concatenate
from keras.optimizers import Adam
from keras.models import Sequential, Model
from sklearn.preprocessing import scale
from keras import backend as K
from keras import optimizers
from keras import regularizers

l1=regularizers.l1(0.0)
l2=regularizers.l2(0.01)
regilization=l1

VERBOSE = 1                     # 0 for turning off logging
#------------------------------------------------------------------------
# stacked auto encoder (sae)
#------------------------------------------------------------------------
# SAE_ACTIVATION = 'tanh'
SAE_ACTIVATION = 'relu'
SAE_BIAS = False
SAE_OPTIMIZER = 'adam'
SAE_LOSS = 'mse'
#------------------------------------------------------------------------
# classifier
#------------------------------------------------------------------------
CLASSIFIER_ACTIVATION = 'relu'
CLASSIFIER_BIAS = False
CLASSIFIER_OPTIMIZER = 'adam'
CLASSIFIER_LOSS = 'binary_crossentropy'
hidden_nodes=64


def param():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help="ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="random seed",
        default=0,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 20",
        default=50,
        type=int)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 10",
        default=10,
        type=int)
    parser.add_argument(
        "-T",
        "--training_ratio",
        help="ratio of training data to overall data: default is 0.90",
        default=0.9,
        type=float)
    parser.add_argument(
        "-S",
        "--sae_hidden_layers",
        help=
        "comma-separated numbers of units in SAE hidden layers; default is '256,128,64,128,256'",
        default='256,128,'+str(hidden_nodes)+',128,256',
        type=str)
    parser.add_argument(
        "-C",
        "--classifier_hidden_layers",
        help=
        "comma-separated numbers of units in classifier hidden layers; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "-D",
        "--dropout",
        help=
        "dropout rate before and after classifier hidden layers; default 0.0",
        default=0.4,
        type=float)
    # parser.add_argument(
    #     "--building_weight",
    #     help=
    #     "weight for building classes in classifier; default 1.0",
    #     default=1.0,
    #     type=float)
    # parser.add_argument(
    #     "--floor_weight",
    #     help=
    #     "weight for floor classes in classifier; default 1.0",
    #     default=1.0,
    #     type=float)
    parser.add_argument(
        "-N",
        "--neighbours",
        help="number of (nearest) neighbour locations to consider in positioning; default is 1",
        default=1,
        type=int)
    parser.add_argument(
        "--scaling",
        help=
        "scaling factor for threshold (i.e., threshold=scaling*maximum) for the inclusion of nighbour locations to consider in positioning; default is 0.0",
        default=0.0,
        type=float)
    args = parser.parse_args()
    return args

def file_open(D,tv_mode):
	#确定数据集的类型
	train_mode,val_mode=tv_mode.split('-',1)
    #print(train_mode+'@'+val_mode)
	train_file='train_val_data/'+D+train_mode+'_train_set.csv'
	train_data=pd.read_csv(train_file)

	val_file='train_val_data/'+D+val_mode+'_val_set.csv'
	val_data=pd.read_csv(val_file)
	#print(train_file);print(val_file)
	choose_mode={'train_set':train_mode,'val_set':val_mode}
	return choose_mode,train_data,val_data

def data_label_seperate(train_df,test_df,batch_size):
	data_size=int(train_df.shape[1]-5)
	len_train=int(train_df.shape[0])
	len_val=int(test_df.shape[0])


	train_AP_features = scale(np.asarray(train_df.iloc[:,0:data_size]).astype(float), axis=1)
	val_AP_features= scale(np.asarray(test_df.iloc[:,0:data_size]).astype(float), axis=1)

	train_Geo_features=train_df.iloc[:,data_size+2:]
	val_Geo_features = test_df.iloc[:,data_size+2:]


	x_all = np.asarray(pd.get_dummies(pd.concat([train_df['Loc_x'], test_df['Loc_x']])))
	y_all = np.asarray(pd.get_dummies(pd.concat([train_df['Loc_y'], test_df['Loc_y']])))

	train_labels = np.concatenate((x_all, y_all), axis=1)
	#print('train_label:',train_labels.shape)
	RSS_train = train_AP_features
	Geo_train=train_Geo_features
	y_train=train_labels[:len_train]

	RSS_val=val_AP_features
	Geo_val=val_Geo_features
	y_val=train_labels[len_train:]
	
	
	Geo_train,Geo_val=geo_preprocess(Geo_train,Geo_val,batch_size)

	

	#trainY = np.concatenate((train_loc_x,train_loc_y),axis = 2)
	#testY = np.concatenate((test_loc_x,test_loc_y),axis = 2)
	#trainY = np.reshape(trainY, (len(trainY),look_back,2))
	return  RSS_train,y_train,RSS_val,y_val,Geo_train,Geo_val,

def geo_preprocess(train_data,val_data,batch_size):
	train_Mag_x=train_data['rGeoX']
	train_Mag_x=np.array(train_Mag_x).astype(float).reshape(-1,1)
	scaler_mag_x=MinMaxScaler()
	train_Mag_x=scaler_mag_x.fit_transform(train_Mag_x)

	train_Mag_y=train_data['rGeoY']
	train_Mag_y=np.array(train_Mag_y).astype(float).reshape(-1,1)
	scaler_mag_y=MinMaxScaler()
	train_Mag_y=scaler_mag_y.fit_transform(train_Mag_y)

	train_Mag_z=train_data['rGeoZ']
	train_Mag_z=np.array(train_Mag_z).astype(float).reshape(-1,1)
	scaler_mag_z=MinMaxScaler()
	train_Mag_z=scaler_mag_z.fit_transform(train_Mag_z)

	train_size=int(len(train_Mag_x))

	val_Mag_x=val_data['rGeoX']
	val_Mag_x=np.array(val_Mag_x).astype(float).reshape(-1,1)
	v_scaler_mag_x=MinMaxScaler()
	val_Mag_x=v_scaler_mag_x.fit_transform(val_Mag_x)

	val_Mag_y=val_data['rGeoY']
	val_Mag_y=np.array(val_Mag_y).astype(float).reshape(-1,1)
	v_scaler_mag_y=MinMaxScaler()
	val_Mag_y=v_scaler_mag_y.fit_transform(val_Mag_y)

	val_Mag_z=val_data['rGeoZ']
	val_Mag_z=np.array(val_Mag_z).astype(float).reshape(-1,1)
	v_scaler_mag_z=MinMaxScaler()
	val_Mag_z=v_scaler_mag_z.fit_transform(val_Mag_z)

	val_size=int(len(val_Mag_x))


	trainX = np.concatenate((train_Mag_x,train_Mag_y,train_Mag_z),axis = 1)
	testX = np.concatenate((val_Mag_x,val_Mag_y,val_Mag_z),axis = 1)

	#print('train_loc_x.shape:',train_loc_x.shape)
	#print('trianY:',trainX.shape)
	
	lengthTrain = len(trainX)
	lengthTest = len(testX)
	while(lengthTrain % batch_size != 0):
		lengthTrain -= 1
	while(lengthTest % batch_size != 0):
		lengthTest -= 1

	return trainX[0:lengthTrain],testX[0:lengthTest]


def build_model(sae_hidden_layers,INPUT_DIM,SAE_ACTIVATION,SAE_BIAS,SAE_OPTIMIZER,SAE_LOSS,batch_size,epochs,VERBOSE,RSS_train,y_train):
    # create a model based on stacked autoencoder (SAE)
	model = Sequential()
	model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
	for units in sae_hidden_layers[1:]:
		model.add(Dense(units, activation=SAE_ACTIVATION, use_bias=SAE_BIAS,activity_regularizer=regilization,))  
	model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS,))
	model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)

        # train the model
	model.fit(RSS_train, RSS_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE)

        # remove the decoder part
	num_to_remove = (len(sae_hidden_layers) + 1) // 2
	for i in range(num_to_remove):
		model.pop()
	model.add(Dropout(dropout))
	return model
	#print(INPUT_DIM)
	#input=Input(batch_shape = (batch_size,INPUT_DIM))
	#x=Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS)(input)
		
	#for units in sae_hidden_layers[1:]:
	#	x=Dense(units, activation=SAE_ACTIVATION, use_bias=SAE_BIAS)(x) 
	#x=Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS)(x)
	#model = Model(inputs = input, outputs = x)

	#model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)
	#model.fit(RSS_train, RSS_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE)

 ##       # remove the decoder part
	#num_to_remove = (len(sae_hidden_layers) + 1) // 2
	#for i in range(num_to_remove):
	#	model.pop()
	return model

    # train the model
	
	
def build_model_MLP(INPUT_DIM,feature_size = 3):
	Geo_input = Input(batch_shape = (batch_size,feature_size))
	AE_output = Input(batch_shape = (batch_size,hidden_nodes))
	
	MLP_input = Concatenate(axis=1)([Geo_input ,AE_output])
	x= MLP_input
	for units in classifier_hidden_layers:
		x=Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=CLASSIFIER_BIAS,activity_regularizer=regilization,)(x)
		x=Dropout(dropout)(x)
	output=Dense(OUTPUT_DIM, activation='sigmoid', use_bias=CLASSIFIER_BIAS,)(x)
	
	model = Model(inputs = [Geo_input ,AE_output], outputs = output)
	#model.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS, metrics=['accuracy'])
	return model

def AE_MLP_combine(model_AE,model_MLP,feature_size = 3):
	RSS_train=Input(shape= (INPUT_DIM,))
	Geo_input = Input(shape= (feature_size,))

	AE_out=model_AE(RSS_train)
	MLP_out=model_MLP(inputs=[Geo_input,AE_out])

	model_AE_MLP = Model(outputs = MLP_out,inputs =[RSS_train,Geo_input])
	model_AE_MLP.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS,metrics=['accuracy'])
	
#	
	return model_AE_MLP

#def LSTM_build(INPUT_DIM,hidden_nodes=hidden_nodes, time_steps = look_back, feature_size = 3):

#	Geo_input = Input(batch_shape = (batch_size,look_back,feature_size))
#	cnn_out=Input(batch_shape = (batch_size,hidden_nodes))				   #type:<class 'tensorflow.python.framework.ops.Tensor'>
#	cell_state_init=tf.random_uniform([batch_size,hidden_nodes])		   #type:<class 'tensorflow.python.framework.ops.Tensor'>
	
#	lstm1 = GRU(hidden_nodes, stateful = False, return_sequences=True, return_state=True,dropout=0.2)(Geo_input,cnn_out)
	
#	lstm1 = GRU(hidden_nodes,return_sequences=True,dropout=0.2)(lstm1)
#	lstm1 = TimeDistributed(Dense((OUTPUT_DIM)))(lstm1)
#	model_LSTM = Model(inputs =[Geo_input,cnn_out],outputs = lstm1)
	
#	return model_LSTM


#def CNN_LSTM_combine(model_CNN,model_LSTM,INPUT_DIM,feature_size = 3):
#	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#	RSS_train=Input(shape= (INPUT_DIM,))
#	Geo_input = Input(shape= (look_back,feature_size))
#	cnn_out=model_CNN(RSS_train)
#	output=model_LSTM(inputs=[Geo_input,cnn_out])

#	combine_model = Model(outputs = output,inputs =(RSS_train,Geo_input))
#	combine_model.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS,metrics=['accuracy'])
	
#	return combine_model


	

def predict_evaluate(model,RSS_train,Geo_train,y_train,RSS_val,Geo_val,y_val,file_acc2loss):
	adam = optimizers.Adam(lr=0.001, beta_1=0.99999, beta_2=0.999, epsilon=1e-09)
	reduce_lr = ReduceLROnPlateau(optimizer=adam, monitor='val_loss', patience=4,factor=0.9, mode='min')

	history = model.fit([RSS_train,Geo_train], y_train, validation_data=([RSS_val,Geo_val], y_val), batch_size=batch_size, epochs=epochs, verbose=VERBOSE,callbacks=[reduce_lr])

	train_result=pd.DataFrame(history.history)
	train_result.to_csv(file_acc2loss)
	del reduce_lr;del adam


if __name__=="__main__":
	args=param()
    # set variables using command-line arguments
	gpu_id = args.gpu_id
	random_seed = args.random_seed
	epochs = args.epochs
	batch_size = args.batch_size
	training_ratio = args.training_ratio
	sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
	if args.classifier_hidden_layers == '':
		classifier_hidden_layers = ''
	else:
		classifier_hidden_layers = [int(i) for i in (args.classifier_hidden_layers).split(',')]
	dropout = args.dropout
	# building_weight = args.building_weight
    # floor_weight = args.floor_weight
	N = args.neighbours
	scaling = args.scaling
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	if gpu_id >= 0:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = ''
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

	


	Epoch=str(epochs)
	delete=['0','1','2']
	for D in delete:
			if D!='2':
				all_mode=[
					'a-a','a-b','a-ab',
					'b-a','b-b','b-ab',
					'ab-a','ab-b','ab-ab'
					]
			else:
				all_mode=['a-a','b-b']
			for mode in all_mode:
		



				file='dropout_0.4'
				

				choose_mode,train_df,test_df=file_open(D,mode)
				file_acc2loss = 'result/'+file+'/'+D+choose_mode['train_set']+'-'+choose_mode['val_set']+'_Epochs_'+Epoch+'Droupout_'+str(dropout)+'_acc.csv'
				print(file_acc2loss)
	
	

				RSS_train,y_train,RSS_val,y_val,Geo_train,Geo_val=data_label_seperate(train_df,test_df,batch_size)
				INPUT_DIM=RSS_train.shape[1]
				OUTPUT_DIM = y_val.shape[1]
	
				model_AE=build_model(sae_hidden_layers,INPUT_DIM,SAE_ACTIVATION,SAE_BIAS,SAE_OPTIMIZER,SAE_LOSS,batch_size,epochs,VERBOSE,RSS_train,y_train)
	
	
				model_MLP=build_model_MLP(INPUT_DIM)

				model_AE_MLP=AE_MLP_combine(model_AE,model_MLP)
				#model_LSTM=LSTM_build(INPUT_DIM)
				#combine_model=CNN_LSTM_combine(model_CNN,model_LSTM,INPUT_DIM)
	
				#model_AE_MLP.fit([RSS_train,Geo_train], y_train, validation_data=([RSS_val,Geo_val], y_val), batch_size=batch_size, epochs=1, verbose=VERBOSE)
				predict_evaluate(model_AE_MLP,RSS_train,Geo_train,y_train,RSS_val,Geo_val, y_val,file_acc2loss)
	
				del model_AE_MLP
				del	model_AE
				del model_MLP







def model_complete(model,dropout,classifier_hidden_layers,CLASSIFIER_ACTIVATION,CLASSIFIER_BIAS,OUTPUT_DIM,CLASSIFIER_OPTIMIZER,CLASSIFIER_LOSS,batch_size,epochs,VERBOSE):
	print("Part 2: buidling a complete model ...")
    # append a classifier to the model
    # class_weight = {
    #     0: building_weight, 1: building_weight, 2: building_weight,  # buildings
    #     3: floor_weight, 4: floor_weight, 5: floor_weight, 6:floor_weight, 7: floor_weight  # floors
    # }
	model.add(Dropout(dropout))
	for units in classifier_hidden_layers:
        #units=128;units2=128
		model.add(Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=CLASSIFIER_BIAS))
		model.add(Dropout(dropout))
	model.add(Dense(OUTPUT_DIM, activation='sigmoid', use_bias=CLASSIFIER_BIAS)) # 'sigmoid' for multi-label classification
	model.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS, metrics=['accuracy'])
	return model