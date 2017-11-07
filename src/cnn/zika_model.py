#################################################################

def zika_model(input_shape=(128, 128,3),weight_path=None):
        model = Sequential()
        model.add(BatchNormalization(input_shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))        
        model.add(MaxPooling2D(pool_size=(2, 2)))
       
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

 
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
	return model        


#################################################################

def zika_Model_VGG16(input_shape=(128, 128,3),weight_path=None):
    from keras.applications.vgg16 import VGG16
    base_model=VGG19(include_top=False,
                   weights='imagenet',
                   input_shape=input_shape)

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(base_model)
    model.add(Flatten())
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
