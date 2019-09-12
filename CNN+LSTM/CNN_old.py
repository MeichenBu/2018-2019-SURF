import argparse
import datetime
import os
import math
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import scale
from timeit import default_timer as timer
import csv

### global constant variables
#------------------------------------------------------------------------
# general
#------------------------------------------------------------------------
                #  number of APs
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
#------------------------------------------------------------------------
# input files
#------------------------------------------------------------------------
if True:
	mode=['a-a','a-b','a-ab','b-a','b-b','b-ab','ab-a','ab-b','ab-ab']
else:
	mode=['ab-ab']

#------------------------------------------------------------------------
# output files
#------------------------------------------------------------------------
path_base = '../results/' + os.path.splitext(os.path.basename(__file__))[0]
path_out =  path_base + '_out'
path_sae_model = path_base + '_sae_model.hdf5'

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
        default='256,128,64,128,256',
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
        default=0.5,
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

def build_model(sae_hidden_layers,INPUT_DIM,SAE_ACTIVATION,SAE_BIAS,SAE_OPTIMIZER,SAE_LOSS,batch_size,epochs,VERBOSE,x_train,y_train,path_sae_model):
    print("\nPart 1: buidling an SAE encoder ...")
    if False:
    # if os.path.isfile(path_sae_model) and (os.path.getmtime(path_sae_model) > os.path.getmtime(__file__)):
        model = load_model(path_sae_model)
    else:
        # create a model based on stacked autoencoder (SAE)
        model = Sequential()
        model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
        for units in sae_hidden_layers[1:]:
            model.add(Dense(units, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))  
        model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
        model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)

        # train the model
        model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE)

        # remove the decoder part
        num_to_remove = (len(sae_hidden_layers) + 1) // 2
        for i in range(num_to_remove):
            model.pop()
    return model

def model_complete(model,dropout,classifier_hidden_layers,CLASSIFIER_ACTIVATION,CLASSIFIER_BIAS,OUTPUT_DIM,CLASSIFIER_OPTIMIZER,CLASSIFIER_LOSS,x_train, y_train,x_val, y_val,batch_size,epochs,VERBOSE):
	print("\nPart 2: buidling a complete model ...")
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

def file_open(delete,tv_mode):
	#确定数据集的类型
	train_mode,val_mode=tv_mode.split('-',1)
    #print(train_mode+'@'+val_mode)
	D= '0'if delete else '1'
	train_file=[D+train_mode+'_train_set.csv']
	train_file="".join(train_file)
	train_data=pd.read_csv(train_file)

	val_file=[D+val_mode+'_val_set.csv']
	val_file="".join(val_file)
	val_data=pd.read_csv(val_file)
	#print(train_file);print(val_file)
	choose_mode={'train_set':train_mode,'val_set':val_mode}
	return choose_mode,train_data,val_data

def data_label_seperate(train_df,test_df):
	data_size=int(train_df.shape[1]-2)
	len_train=int(train_df.shape[0])
	len_val=int(test_df.shape[0])


	train_AP_features = scale(np.asarray(train_df.iloc[:,0:data_size]).astype(float), axis=1)
	val_AP_features= scale(np.asarray(test_df.iloc[:,0:data_size]).astype(float), axis=1)

	x_all = np.asarray(pd.get_dummies(pd.concat([train_df['Loc_x'], test_df['Loc_x']])))
	y_all = np.asarray(pd.get_dummies(pd.concat([train_df['Loc_y'], test_df['Loc_y']])))

	train_labels = np.concatenate((x_all, y_all), axis=1)
	#print('train_label:',train_labels.shape)
	x_train = train_AP_features
	y_train=train_labels[:len_train]

	x_val=val_AP_features
	y_val=train_labels[len_train:]

	return x_train,y_train,x_val,y_val

def predict_evaluate(model,x_train,y_train,x_val,y_val,file_structure,file_acc2loss):
	for i in range(epochs):
		history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=1, verbose=VERBOSE) #validation_split=0.1, validation_data=(test_x, test_y)
		# need to reset state for every epoch
		model.reset_states()
		#print('hidden_state:',hidden_state)
		# list all data in history
		'''
		print('history.keys()',hist.history.keys())
		# summarize history for accuracy
		plt.plot(hist.history['acc'])
		plt.plot(hist.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		'''
		print('Real Epoches:',i+1)
		with open(file_acc2loss,'a', newline='') as csvfile:
			if not os.path.getsize(file_acc2loss):        #file is empty
				spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
				spamwriter.writerow(['epochs','loss','acc','val_loss','val_acc'])#, 'val_loss','val_acc' 
				
			data = ([
				i,history.history['loss'][0],history.history['acc'][0], history.history['val_loss'][0], history.history['val_acc'][0]
				])  
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow(data)


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
#参数设置结束
	### initialize random seed generator of numpy
    ####设置
	np.random.seed(random_seed)
    #--------------------------------------------------------------------
    # import keras and its backend (e.g., tensorflow)
    #--------------------------------------------------------------------
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	if gpu_id >= 0:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = ''
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # supress warning messages
	import tensorflow as tf
	tf.set_random_seed(random_seed)  # initialize random seed generator of tensorflow
	from keras.layers import Dense, Dropout
	from keras.models import Sequential, load_model

	DELE=[True,False]
	Epoch=str(epochs)
	for delete in DELE:
		D= '0'if delete else '1'
		for tv_mode	 in mode:
			choose_mode,train_df,test_df=file_open(delete,tv_mode)

			file_structure = 'result/'+D+choose_mode['train_set']+'-'+choose_mode['val_set']+'_'+'model_ts=30_256_5_100.png'
			file_acc2loss = 'result/'+D+choose_mode['train_set']+'-'+choose_mode['val_set']+'_Epochs_'+Epoch+'Droupout_'+str(dropout)+'_acc.csv'

			print(file_acc2loss)
			x_train,y_train,x_val,y_val=data_label_seperate(train_df,test_df)
			#print(x_train.shape)
			#print(y_train.shape)
			#print(x_val.shape)
			#print(y_val.shape)
			#print('@@@@@@@@@@@@@@@@@@@')
			INPUT_DIM=x_train.shape[1]
			OUTPUT_DIM = y_val.shape[1]
			model=build_model(sae_hidden_layers,INPUT_DIM,SAE_ACTIVATION,SAE_BIAS,SAE_OPTIMIZER,SAE_LOSS,batch_size,epochs,VERBOSE,x_train,y_train,path_sae_model)
			model=model_complete(model,dropout,classifier_hidden_layers,CLASSIFIER_ACTIVATION,CLASSIFIER_BIAS,OUTPUT_DIM,CLASSIFIER_OPTIMIZER,CLASSIFIER_LOSS,x_train, y_train,x_val, y_val,batch_size,epochs,VERBOSE)


			file_structure = 'result/'+D+choose_mode['train_set']+'-'+choose_mode['val_set']+'_'+'model_ts=30_256_5_100.png'
			file_acc2loss = 'result/'+D+choose_mode['train_set']+'-'+choose_mode['val_set']+'_Epochs_'+Epoch+'Droupout_'+str(dropout)+'_acc.csv'
			

			predict_evaluate(model,x_train,y_train,x_val,y_val,file_structure,file_acc2loss)
			del model