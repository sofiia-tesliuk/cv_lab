from cnn.networks.lenet import LeNet
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
                help="(optional) path to weights file")
ap.add_argument("-o", "--sgd-optimizer", type=int, default=-1,
                help="(optional) sgd or rmsp")
ap.add_argument("-a", "--adadelta-optimizer", type=int, default=-1,
                help="(optional) sgd or rmsp")
ap.add_argument("-g", "--gray-scale", type=int, default=-1,
                help="(optional) if make images gray-scale for training")
args = vars(ap.parse_args())

print('[INFO] loading data.')
trainData = np.load('data/images_train.npy')
testData = np.load('data/images_test.npy')
trainLabels = np.load('data/labels_train.npy')
testLabels = np.load('data/labels_test.npy')

depth_size = 3

if args["gray_scale"] == 1:
    trainData = np.dot(trainData[...,:3], [0.299, 0.587, 0.114])
    testData = np.dot(testData[...,:3], [0.299, 0.587, 0.114])
    depth_size = 1
    trainData = trainData.reshape((trainData.shape[0], 32, 32, depth_size))
    testData = testData.reshape((testData.shape[0], 32, 32, depth_size))

# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    trainData = trainData.reshape((trainData.shape[0], depth_size, 32, 32))
    testData = testData.reshape((testData.shape[0], depth_size, 32, 32))

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0


# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 2)
testLabels = np_utils.to_categorical(testLabels, 2)


print("[INFO] compiling model...")

if args["sgd_optimizer"] == 1:
    opt = SGD(lr=0.01)
elif args["adadelta_optimizer"] == 1:
    opt = Adadelta(lr=0.01)
else:
    opt = RMSprop(lr=0.01)

model = LeNet.build(numChannels=depth_size, imgRows=32, imgCols=32,
                    numClasses=2,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(trainData, trainLabels, batch_size=128, epochs=20,
              verbose=1)

    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
                                      batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)


# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # extract the image from the testData if using "channels_first"
    # ordering
    if K.image_data_format() == "channels_first":
        image = (testData[i][0] * 255).astype("uint8")

    # otherwise we are using "channels_last" ordering
    else:
        image = (testData[i] * 255).astype("uint8")

    if depth_size == 1:
        # merge the channels into one image
        image = cv2.merge([image] * 3)

    # resize the image from a 28 x 28 image to a 96 x 96 image so we
    # can better see it
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

    p_or_b = lambda x: 'Bird' if x == 1 else 'Plane'

    # show the image and prediction
    cv2.putText(image, p_or_b(prediction[0]), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
                                                    np.argmax(testLabels[i])))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)

