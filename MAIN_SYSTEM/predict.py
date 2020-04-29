from keras.preprocessing import image

def do_prediction():
    img_path = 'images/my_image.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    my_image = scipy.misc.imread(img_path)
    imshow(my_image)
    print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
    print(model.predict(x))