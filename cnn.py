import tensorflow as tf
import tflearn
import cv2
import os
import numpy as np

#used on resize
imageDimension = 56


#links all the images in one list
def getImagesInFolder(folderPath):
    listOfImages = []
    listOfFiles = os.listdir(folderPath)
    for eachFile in listOfFiles:
        if((".jpg" in eachFile) or (".png" in eachFile) or (".bmp" in eachFile)):
            print(folderPath + "/" + eachFile)
            eachImage = cv2.imread(folderPath + "/" + eachFile, 1)
            listOfImages.append(cv2.resize(eachImage, (imageDimension, imageDimension)))
    return listOfImages

#creates array
def createVectorizedArrayOfImages(listOfImages):
    vectorizedArray = None
    for eachImage in listOfImages:
        if(vectorizedArray == None):
            vectorizedArray = eachImage.flatten()
        else:
            vectorizedArray = np.vstack((vectorizedArray, eachImage.flatten()))
    return vectorizedArray


def call(fpointer):
    
    #actually gets the images from the directory training and validation
    listOfTrainImages = getImagesInFolder("estilos/training/Bright")
    listOfValidImages = getImagesInFolder("estilos/validation/Bright")

    totalOfBrightTrainImages = len(listOfTrainImages)
    totalOfBrightValidImages = len(listOfValidImages)
    totalOfNoirTrainImages = len(getImagesInFolder("estilos/training/Noir"))
    totalOfNoirValidImages = len(getImagesInFolder("estilos/validation/Noir"))
    totalOfLongExposureTrainImage = len(getImagesInFolder("estilos/training/LongExposure"))
    totalOfLongExposureValidImage = len(getImagesInFolder("estilos/validation/LongExposure"))
    totalOfMinTrainImage = len(getImagesInFolder("estilos/training/Minimalist"))
    totalOfMinValidImage = len(getImagesInFolder("estilos/validation/Minimalist"))

    listOfTrainImages.extend(getImagesInFolder("estilos/training/Noir"))
    listOfTrainImages.extend(getImagesInFolder("estilos/training/LongExposure"))
    listOfTrainImages.extend(getImagesInFolder("estilos/training/Minimalist"))
    listOfValidImages.extend(getImagesInFolder("estilos/validation/Noir"))
    listOfValidImages.extend(getImagesInFolder("estilos/validation/LongExposure"))
    listOfValidImages.extend(getImagesInFolder("estilos/validation/Minimalist"))

    vectorizedArrayOfTrainImages = createVectorizedArrayOfImages(listOfTrainImages)
    vectorizedArrayOfValidImages = createVectorizedArrayOfImages(listOfValidImages)

    # Parameters
    learning_rate = 0.0001
    training_iters = 200000
    batch_size = 56
    display_step = 10
    #saving for tensorBoard
    model_visualStyle = 'class4Estilos-{}-{}.model'.format(learning_rate,'2conv')
    # Network Parameters / 3 is the number of color channels
    n_input = imageDimension * imageDimension * 3 # MNIST data input (img shape: 28*28)
    n_classes = 4 # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units
    

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, imageDimension, imageDimension, 3])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=3)
        
        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv2, k=3)
       
        # Convolution Layer
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        # Max Pooling (down-sampling)
        conv3 = maxpool2d(conv3, k=3)      

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input

        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 3 input channels, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 16])),
        
        # 5x5 conv, 3 input channels, 32 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        
        'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64])),

        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([16])),
        'bc2': tf.Variable(tf.random_normal([32])),
        'bc3': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    min_test_loss = 999999.0

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations

        train_xdata = vectorizedArrayOfTrainImages
        test_xdata = vectorizedArrayOfValidImages

        ##each array represents one image style
        train_labels = np.empty([0, 4])
        for i in range(totalOfBrightTrainImages):
            train_labels = np.vstack((train_labels, [0, 1, 0, 0]))

        for i in range(totalOfNoirTrainImages):
            train_labels = np.vstack((train_labels, [1, 0, 0, 0]))
        
        for i in range(totalOfLongExposureTrainImage):
            train_labels = np.vstack((train_labels, [0, 0, 1, 0]))
            
        for i in range(totalOfMinTrainImage):
            train_labels = np.vstack((train_labels, [1, 1, 1, 1]))
            
        #same on testing
        test_labels = np.empty([0, 4])
        for i in range(totalOfBrightValidImages):
            test_labels = np.vstack((test_labels, [0, 1, 0, 0]))

        for i in range(totalOfNoirValidImages):
            test_labels = np.vstack((test_labels, [1, 0, 0, 0]))

        for i in range(totalOfLongExposureValidImage):
            test_labels = np.vstack((test_labels, [0, 0, 1, 0]))
            
        for i in range(totalOfMinValidImage):
            test_labels = np.vstack((test_labels, [1, 1, 1, 1]))
            
        while step < training_iters:
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            rand_index = np.random.choice(len(train_xdata), size=batch_size)
            #print(len(train_xdata))

            rand_xdata = train_xdata[rand_index]
            #print("rand_xdata", rand_xdata)
   
            rand_labels = train_labels[rand_index]
        
            batch_x = rand_xdata
            batch_y = rand_labels
        
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            if step % display_step == 0:
        
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})

                test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: test_xdata, y: test_labels, keep_prob: 1.})

                if(1):
                    #fpointer.write("Iter " + str(step*1) + ", Minibatch Loss= " + \
                    #    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    #    "{:.5f}".format(acc))
                    print("Iter " + str(step*1) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Training Accuracy= " + \
                        "{:.5f}".format(acc))

                    # Calculate accuracy for 256 mnist test images
                    #fpointer.write("Testing Accuracy:" + str(test_loss) +" " + str(test_acc) + "\n")
                    print("Testing Accuracy:" + str(test_loss) +" " + str(test_acc))

                    #if(test_acc > 0.9):
                    #    break

                    min_test_loss = test_loss

            step += 1
            #saving for tensorBoard
            #model.save = (model_visualStyle)
        print("Optimization Finished!")
#running the code
call(0)
