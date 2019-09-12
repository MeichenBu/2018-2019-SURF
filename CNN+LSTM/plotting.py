import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

val_acc='val_acc'
val_loss='val_loss'
acc='acc'
loss='loss'

delete=['0','1','2']
file='dropout_0.4'
epoch=50
dropout_rang=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]		  #0.4
dropout_select=[4,4,4,4,4,4,4,4,4]
Epochs=str(epoch)


x=range(epoch)
all_mode=['a-a','a-b','a-ab','b-a','b-b','b-ab','ab-a','ab-b','ab-ab']
		#[  0,    1,    2,     3,    4,    5,     6,      7,     8]
select=[  6,6]
DELETE_type=[0,1,0,0,0,0,0,0,0]
#dropout=[0.0,0.5]
if __name__=="__main__":
	for num in range(len(select)):
		mode=all_mode[select[num]]
		D=delete[DELETE_type[num]]
		dropout=dropout_rang[dropout_select[num]]
		train_mode,val_mode=mode.split('-',1)
		file_name='result/'+file+'/'+D+train_mode+'-'+val_mode+'_Epochs_'+Epochs+'Droupout_'+str(dropout)+'_acc.csv'
		train_result=pd.read_csv(file_name)
		y=train_result[val_acc]
		y=y[y.shape[0]-50:]
		plt.plot(x,y,label=train_mode+'-'+val_mode+''+D+'_Dropout_'+str(dropout))

	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')

	plt.legend(loc = 0, ncol = 1)
	plt.title('Accuracy via Epochs with First methods',fontsize='large',fontweight='bold')
	
	plt.show()

