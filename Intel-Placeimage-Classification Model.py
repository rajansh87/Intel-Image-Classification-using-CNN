from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator

test_path='seg_test/'
train_path='seg_train/'
num_of_epochs=1
s_p_epoch=2300
v_steps=12000

def myModel(num_of_epochs,train_path,test_path,s_p_epoch,v_steps):

    model = Sequential()

    #inputlayer : apply filters
    model.add(Convolution2D(filters=32, 
                            kernel_size=(3,3), 
                            activation='relu',
                       input_shape=(64, 64, 3)
                           ))

    # pooling layer where we are doing maxpooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #modification for increasing accuracy
    model.add(Convolution2D(filters=32, 
                            kernel_size=(3,3), 
                            activation='relu',
                           ))

    #modification for increasing accuracy
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #layer inwhich we areconverting 2d/3d image to 1d image i.e flattening
    model.add(Flatten())

    # layer: appling relu to give positive output
    # from here our hidden layerrs starts
    model.add(Dense(units=128, activation='relu'))

    #output layer : to provide binary output using sigmoid function
    model.add(Dense(units=6, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



    #image augmentation

    #url : https://keras.io/api/preprocessing/image/ 
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory(
            train_path,
            target_size=(64,64),
            batch_size=32,
            class_mode='categorical')
    test_set = test_datagen.flow_from_directory(
            test_path,
            target_size=(64,64),
            batch_size=32,
            class_mode='categorical')


    training_set.class_indices # to see classes of our dataset

    history = model.fit(
            training_set,
            steps_per_epoch=s_p_epoch,
            epochs=num_of_epochs,
            validation_data=test_set,
            validation_steps=v_steps)

    finalAccuracy = history.history["accuracy"]
    #model.save("new-cnn-placeimage_model.h5")#save model
    return finalAccuracy



obj=myModel(num_of_epochs,train_path,test_path,s_p_epoch,v_steps)
print(obj)
