

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential, load_model, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, RMSprop, SGD

from keras.applications.vgg16 import VGG16 
import matplotlib.pyplot as plt

from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.xception import preprocess_input, decode_predictions
from keras.applications.densenet import preprocess_input, decode_predictions

from IPython.display import display
from PIL import Image
from sklearn import metrics
from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np
import os

from datetime import datetime

from keras.utils import plot_model
import keras.backend as K
from hyperopt import hp, tpe, fmin, Trials

import pickle
import os
import traceback
import utils




#OPTIMIZER
#My space of choices
space ={
	'batch_size': hp.choice('batch_size', [15, 10, 5]),
	'optimizer': hp.choice('optimizer', ['Adam', 'RMSprop', 'SGD']),
	'activation': hp.choice('activation', ['relu', 'elu']),
	'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3]), 
	'epoch1': hp.choice('epoch1', [30, 50, 70]),    
	'epoch2': hp.choice('epoch2', [60, 80, 100])
	#'steps_per_epoch': hp.choice('steps_per_epoch', 100, 200, 300)    


}

def get_file_list(input_dir):
    return [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]

def get_random_files(file_list, N):
    return random.sample(file_list, N)

def copy_files(random_files, input_dir, output_dir):
    for file in random_files:
        shutil.move(os.path.join(input_dir, file), output_dir)

def main(input_dir, output_dir, N):
    file_list = get_file_list(input_dir)
    random_files = get_random_files(file_list, N)
    copy_files(random_files, input_dir, output_dir)

def optimize(params):
	#TENTATIVA 1
	#net.python3BuildandTrain(model, vgg16, params['dropout'], params['activation'], params['optimizer'], params['epoch1'], params['epoch2']):

	#TENTATIVA 2
	print("DATASET")
	train_data_dir = "RIMONE-db-r2_aug/train"
	validation_data_dir = "RIMONE-db-r2_aug/valid"
	tests_data_dir = "RIMONE-db-r2_aug/test"


	# 600/450 _ 500/375 _ 400/300 _ 300/225
	batch_size = params['batch_size']
	batch_size_val = 4 # if Tensorflow throws a memory error while validating at end of epoch, decrease validation batch size her
	img_width = 128  # Change image size for training here
	img_height = 128 # Change image size for training here
#set data augmentation parameters here
	datagen = ImageDataGenerator(rescale=1., 
		featurewise_center=True,
		rotation_range=10,
		width_shift_range=.1,
		height_shift_range=.1,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		vertical_flip=False,
		fill_mode="reflect")

	# normalization neccessary for correct image input to VGG16
	datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)

	# no data augmentation for validation
	# set data augmentation parameters here
	validgen = ImageDataGenerator(rescale=1., 
		featurewise_center=True,
		rotation_range=10,
		width_shift_range=.1,
		height_shift_range=.1,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		vertical_flip=False,
		fill_mode="reflect")

	validgen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)


	train_gen = datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_height, img_width),
		batch_size=params['batch_size'],
		class_mode="binary",
		shuffle=True, 
		#save_to_dir="_augmented_images/", 
		#save_prefix="aug_"
		)

	val_gen = validgen.flow_from_directory(
		validation_data_dir,
		target_size=(img_height, img_width),
		batch_size=batch_size,
		class_mode="binary",
		shuffle=True)

	# no data augmentation for test
	teste = ImageDataGenerator(rescale=1., featurewise_center=True)
	test_gen = teste.flow_from_directory(
		tests_data_dir,
		target_size=(img_height, img_width),
		batch_size=1,
		class_mode="binary",
		shuffle=False)

#amount of samples
	train_samples = len(train_gen.filenames)
	validation_samples = len(val_gen.filenames)
	test_samples = len(test_gen.filenames)

	print("MODEL")
	vgg16 = VGG16(weights='imagenet', include_top=False)
	#vgg19 = VGG19(weights='imagenet', include_top=False)

	for layer in vgg16.layers:
		layer.trainable = False

	model = Sequential()


	model.add(vgg16)

	model.add(GlobalAveragePooling2D())

	model.add(Dense(128))
	model.add(Dropout(params['dropout']))
	#model.add(BatchNormalization())
	model.add(Activation(params['activation']))
	model.add(Dropout(params['dropout']))
	model.add(Dense(1, activation='sigmoid')) #Saida = 2

	optimizer = params['optimizer']
	checkpoint = ModelCheckpoint('weights_best.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

	callbacks_list = [checkpoint, early_stopping]
	print("TRAIN")
	model.compile(loss='binary_crossentropy',
				optimizer=optimizer,
				metrics=['acc', 'mae'])

	#training
	history = model.fit_generator(epochs=params['epoch1'],
							callbacks=callbacks_list,
							shuffle=True, 
							validation_data=val_gen,
							generator=train_gen,
							steps_per_epoch =300,
		 					validation_steps=100)

	model.save("VGG16.h5")

	for layer in model.layers[:15]:
		layer.trainable = False

	for layer in model.layers[15:]:
		layer.trainable = True
	print("TRAIN 2")
	model.compile(loss='binary_crossentropy',
				optimizer=optimizer,
				metrics=['acc', 'mae'])


	#training
	history = model.fit_generator(epochs=params['epoch2'], 
								callbacks=callbacks_list, 
								shuffle=True, 
								validation_data=val_gen,
								generator=train_gen, 
								steps_per_epoch=300, 
								validation_steps=100) 

	model.save("VGG16.h5")

	print('\nPREDICT\n')

	preds = model.predict_generator(test_gen, len(test_gen))
	preds_rounded = []
	
	print("TESTING")

	for pred in preds:
		if (pred > .5):
			preds_rounded.append("1")
		else:
			preds_rounded.append("0")

	file_list = get_file_list(tests_data_dir + '/false')
	#inicializing float values
	tp = 0.0
	fp = 0.0
	tn = 0.0
	fn = 0.0
	

	for i in range(0, len(file_list)):
		if (preds_rounded[i] == '0'):
			tn=tn+1
		else:
			fp=fp+1

	
	
	file_list = get_file_list(tests_data_dir + '/true')  
	ini = len(file_list)

	for i in range(ini, ini+len(file_list)-1):
		if (preds_rounded[i] == '1'):
			tp=tp+1
		else:
			fn=fn+1

	print(str(tp) + " " + str(tn) + " " + str(fp)+ " " +str(fn))

	predFile = open("preds.txt", "a")

	predFile.write("{}\n".format(preds))

	S = tp/(tp+fn)#sensibility
	P = tp/(tp+fp)#precision
	E = tn/(tn+fp)#specifity
	A = (tp+tn)/(tp+tn+fp+fn)#accuracy
	
	hs = open("log.txt","a")


	hs.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(tp, tn, fp, fn, S, P, E, A))
	
	print('A: ' + str(A) + '\tS: ' + str(S) + '\tP: ' + str(P) + '\tE: ' + str(E)  )
	
	with open('log.csv', 'a+') as LOG:
		result = ",".join([str(x) for x in [params['batch_size'], params['optimizer'], params['activation'], params['epoch1'], params['epoch2'], tp, tn, fp, fn, S, P, E, A]])
		LOG.write(result+"\n")


	return S

if __name__ == '__main__':

	while True:
		best = fmin(
		optimize,
		space,
		algo=tpe.suggest,
		max_evals=100
		) 