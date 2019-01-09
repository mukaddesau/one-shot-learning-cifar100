import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras as ks
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
import math
import cv2
import numpy

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'C:\\Users\\Mukaddes\\Desktop\\crop2\\train_1_1_10_15'
# train_data_dir = 'data/train'
validation_data_dir = train_data_dir


# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def save_bottlebeck_features():
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)
    return num_classes


def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    # plt.figure(1)

    # summarize history for accuracy

    # plt.subplot(211)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    # plt.subplot(212)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


def predict(image_path, conf_mat):
    # load the class_indices saved in the earlier step
   
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below
    

    orig = cv2.imread(image_path)

    # print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    
    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)
  

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label_id_relation = {}
    for item in inv_map:
        lbl_str = inv_map[item]
        label_id_relation[lbl_str] = item

    label = inv_map[inID]

    temp = image_path.split('\\')
    real_label = temp[5]

    conf_mat[label_id_relation[real_label],inID] += 1
        

    # get the prediction label
    print("Image ID: {}, Label: {}, Real Label: {}".format(inID, label,real_label))
    K.clear_session()
    # display the predictions with the image
    # cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                # cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    # cv2.imshow("Classification", orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

num_of_classes = save_bottlebeck_features()
train_top_model()

conf_mat = numpy.zeros(shape=(num_of_classes,num_of_classes))

filename = "C:\\Users\\Mukaddes\\Desktop\\directories.txt"
lines = tuple(open(filename, 'r'))
lines = [x.strip() for x in lines]
for folder in lines:
   # print("folder " + folder)
   true_count = 0
   total_count = 0
   for file in os.listdir(folder):    
       filepath = os.path.join(folder, file)  
       print("filepath" + filepath)  
       predict(filepath,conf_mat)    
       
print(numpy.matrix(conf_mat))

total_sum = numpy.sum(conf_mat)
print("total: " + str(total_sum))

b = numpy.asarray(conf_mat)
diagonal_sum = numpy.trace(b)
print ('Diagonal (sum): ', numpy.trace(b))

acc = diagonal_sum / total_sum
print("accuracy: " + str(acc))

print(train_data_dir)
# for folder in folderList:
#     print(folder)
#     true_count = 0
#     total_count = 0
#     tmp = predict(folder)
#     if(tmp):
#         true_count = true_count + 1
#     total_count = total_count + 1
        
# import glob   
# path = 'C:\\Users\\Mukaddes\\Desktop\\cifar-100 testset\\keyboard\\*.png'   
# files=glob.glob(path) 
# for file in files:
#     print("{}",file)  
# for file in files: 
#     print("{}", file)   
#     tmp = predict(file)
#     print("forda")
#     if(tmp):
#         true_count = true_count + 1
#     total_count = total_count + 1



cv2.destroyAllWindows()