from keras.models import load_model
model=load_model("cnn-intel-image-model.h5")  #load model  <- this has run on 3 epochs with ~85% accuracy


from keras.preprocessing import image


test_image = image.load_img("seg_pred/14.jpg",target_size=(64,64))


#test_image #since this format is PIL or pillow so it can be printed

test_image = image.img_to_array(test_image)  #convert PIL image to numpy array


import numpy as np


test_image = np.expand_dims(test_image,axis=0)
#since keras uses tensor flow and for tensorflow it needs 4d image so we converted 3d image to 4d image using above

result = model.predict(test_image)

print("\n\n\n\nPredictions : ",end="")
if result[0][0]==1:
    print("Buildings")
elif result[0][1]==1:
    print("Forest")
elif result[0][2]==1:
    print("Glacier")
elif result[0][3]==1:
    print("Mountain")
elif result[0][4]==1:
    print("Sea")
else:
    print("Street")



