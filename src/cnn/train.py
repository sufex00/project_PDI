#################################################################

def Train(input_shape = (128, 128, 1)):
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image

#X_train = dataset[:, 1:785]
#y_train = dataset[:, 0]



# normalize inputs from 0-255 to 0-1

	#x_train = x_train / 127

	#x_test = x_test / 127

# one hot encode outputs
	#y_train = np_utils.to_categorical(y_train)
	#y_test = np_utils.to_categorical(y_test)

	#x_train = np.asarray(x_train)
	#y_train = np.asarray(y_train)

	x_t = []
	y_t = []
 
	#x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')

    	#x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') 

	h = KFold_Train(x_train,y_train)
	return h
     
'''
	for idx in range(x_train.shape[0]):
		x_t.append(x_train[idx])
		y_t.append(y_train[idx])
		aux_x_t = x_t[idx]
		aux_y_t = y_t[idx]
		flipped_img=cv2.flip(x_t[idx],1)
		rows,cols,channel = input_shape
		x_t.append(flipped_img)
		y_t.append(y_train[idx])

		for rotate_degree in [90,180,270]:
		    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_degree,1)
		    dst = cv2.warpAffine(aux_x_t,M,(cols,rows))
		    x_t.append(dst)
		    y_t.append(aux_y_t)
		    
		    dst = cv2.warpAffine(flipped_img,M,(cols,rows))
		    x_t.append(dst)
		    y_t.append(aux_y_t)
	
	y_train = np.array(y_t, np.uint8)
	x_train = np.array(x_t, np.uint8)
'''
	#x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')

	#x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

	#a =np.empty((x_train.shape[0],28, 28, 3))
	#a[:][:][:][:]=x_train[:][:][:][:][:]
	#x_train = a

	#KFold_Train(x_train,y_train)

#################################################################

def KFold_Train(x_train,y_train,nfolds=3,batch_size=128):
	model = zika_model(input_shape)
	kf = KFold(n_splits=nfolds, shuffle=True, random_state=1)
	num_fold = 0 
	for train_index, test_index in kf.split(x_train, y_train):
		start_time_model_fitting = time.time()
		X_train = x_train[train_index]
		Y_train = y_train[train_index]
		X_valid = x_train[test_index]
		Y_valid = y_train[test_index]
		
		#X_train = x_test.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
		
		#X_valid = x_test.reshape(X_valid.shape[0], 1, 28, 28).astype('float32')
		model = zika_model(input_shape)
		num_fold += 1
		print('Start KFold number {} from {}'.format(num_fold, nfolds))
		print('Split train: ', len(X_train), len(Y_train))
		print('Split valid: ', len(X_valid), len(Y_valid))
		
		kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

		epochs_arr =  [30, 15, 10]
		learn_rates = [0.001, 0.0001, 0.00001]

		for learn_rate, epochs in zip(learn_rates, epochs_arr):
		    print('Start Learn_rate number {} from {}'.format(epochs,learn_rate))
		    opt  = optimizers.Adam(lr=learn_rate)
		    model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
		                  optimizer=opt,
		                  metrics=['accuracy'])
		    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1),
		    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

		    history = model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),
		          batch_size=32,verbose=1, epochs=epochs,callbacks=callbacks,shuffle=True)
		
		if os.path.isfile(kfold_weights_path):
		    model.load_weights(kfold_weights_path)
		
		p_valid = model.predict(X_valid, batch_size = 32, verbose=2)
		return history

#################################################################



