#!coding=utf-8
import tensorflow as tf
import numpy as np
import json
import utils
import cv2
import time
import os

class Vgg16:

    def __init__(self):
        self.loaded = False
        self.width = 228
        self.height = 228
        self.params = []

    def loadWithUntrainedJson(self, srcDir):
    # Todo: load train information with a JSON file
        jsonFile = os.path.join(srcDir, "build.json")
        data = utils.readJsonFromFile(jsonFile)

        self.learningRate = float(data['learning_rate'])
        self.momentum = float(data['momentum'])
        self.batchsize = int(data['batchsize'])
        self.batches = int(data['batches'])
        self.channel = int(data['channel'])
        self.classes = int(data['classes'])

        self.trainList = os.path.join(srcDir, str(data['trainlist']))
        self.labelList = os.path.join(srcDir,str(data['labellist']))
        self.classnameList = str(data['classnamelist'])
        self.loaded = True
        

    def loadWithTrainedJson(self, jsonFile):
    # Todo: load train information with a JSON file
        self.trained = True

    def train(self, tgtDir):
        self.buildNet()
        init = tf.global_variables_initializer()

        trainSet = open(self.trainList).readlines()
        initlabel = open(self.labelList).readlines()
        label = [int(x) for x in initlabel]

        with tf.Session() as sess:
            sess.run(init)
            for batchIndex in range(self.batches):
                startTime = time.time()
                imageSet = utils.randomChoose(trainSet, self.batchsize)
                x = utils.loadImagesFromFile(imageSet, self.width, self.height)
                imageLabel = utils.getLabels(trainSet, imageSet, label)

                y = []
                for i in range(self.batchsize):
                    y.append([1 if j==imageLabel[i] else 0 for j in range(self.classes)])
                sess.run(self.optimizer, feed_dict={
                    self.x: x,
                    self.y: y
                    })
                endTime = time.time()

                curLoss = sess.run(self.loss, feed_dict={
                    self.x: x,
                    self.y: y
                    })
                print("Batch #%d processing time %.2fs, loss = %.5f"%(batchIndex+1, endTime-startTime, curLoss))

                if batchIndex % 10 == 0:
                    correctImage = 0.0
                    for i in range(self.batchsize):
                        predict = sess.run(self.prediction, feed_dict={
                            self.x: [x[i]],
                            self.y: [[1 if j==imageLabel[i] else 0 for j in range(self.classes)]]
                            })
                        if predict == imageLabel[i]:
                            print predict
                            correctImage += 1
                    print("Valid in step %d: recall: %.5f"%(batchIndex+1, correctImage/self.batchsize))


            self.saveAll(tgtDir)

    def predict(self, imageListFile):

    def saveAll(self, tgtDir):
        #generate json file
        modelJson = {}
        modelJson['width'] = self.width
        modelJson['height'] = self.height
        modelJson['batches'] = self.batches
        modelJson['batchsize'] = self.batchsize
        modelJson['learningRate'] = self.learningRate
        modelJson['classes'] = self.classes
        modelJson['classnameList'] = self.classnameList
        jsonFile = os.path.join(tgtDir, 'model.json')
        json.dump(modelJson, open(jsonFile, 'w'))

        modelFile = os.path.join(tgtDir, 'model.npy')
        param = []
        for each in self.params:
            param.append(np.array(each.eval()))
        param = np.array(param)
        np.save(modelFile, param)
        return
        

    # Build the basic structure of network
    def buildNet(self):
        self.x = tf.placeholder(tf.float32, shape = (None, self.width, self.height, self.channel), name= 'input_layer')
        self.y = tf.placeholder(tf.int32, shape = (None, self.classes), name = 'labels')

        # Conv1
        with tf.name_scope('conv1_1') as scope:
            kernel = self.getWeight([3,3,3,64])
            bias = self.getBias([64])
            self.params += [kernel, bias]
            self.conv1_1 = tf.nn.relu(self.con2d(self.x, kernel)+bias, name = scope)

        with tf.name_scope('conv1_2') as scope:
            kernel = self.getWeight([3,3,64,64])
            bias = self.getBias([64])
            self.params += [kernel, bias]
            self.conv1_2 = tf.nn.relu(self.con2d(self.conv1_1, kernel)+bias, name = scope)

        self.maxpool1 = tf.nn.max_pool(self.conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool1')

        with tf.name_scope('conv2_1') as scope:
            kernel = self.getWeight([3,3,64,128])
            bias = self.getBias([128])
            self.params += [kernel, bias]
            self.conv2_1 = tf.nn.relu(self.con2d(self.maxpool1,kernel)+bias, name=scope)

        with tf.name_scope('conv2_2') as scope:
            kernel = self.getWeight([3,3,128,128])
            bias = self.getBias([128])
            self.params += [kernel, bias]
            self.conv2_2 = tf.nn.relu(self.con2d(self.conv2_1, kernel)+bias, name=scope)

        self.maxpool2 = tf.nn.max_pool(self.conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool2')

        # conv3
        with tf.name_scope('conv3_1') as scope:
            kernel = self.getWeight([3, 3, 128, 256])
            bias = self.getBias([256])
            self.params += [kernel, bias]
            self.conv3_1 = tf.nn.relu(self.con2d(self.maxpool2, kernel) + bias, name=scope)

        with tf.name_scope('conv3_2') as scope:
            kernel = self.getWeight([3, 3, 256, 256])
            bias = self.getBias([256])
            self.params += [kernel, bias]
            self.conv3_2 = tf.nn.relu(self.con2d(self.conv3_1, kernel) + bias, name=scope)

        with tf.name_scope('conv3_3') as scope:
            kernel = self.getWeight([3, 3, 256, 256])
            bias = self.getBias([256])
            self.params += [kernel, bias]
            self.conv3_3 = tf.nn.relu(self.con2d(self.conv3_2, kernel) + bias, name=scope)

        self.maxpool3 = tf.nn.max_pool(self.conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool3')

        # conv4
        with tf.name_scope('conv4_1') as scope:
            kernel = self.getWeight([3, 3, 256, 512])
            bias = self.getBias([512])
            self.params += [kernel, bias]
            self.conv4_1 = tf.nn.relu(self.con2d(self.maxpool3, kernel) + bias, name=scope)

        with tf.name_scope('conv4_2') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            self.params += [kernel, bias]
            self.conv4_2 = tf.nn.relu(self.con2d(self.conv4_1, kernel) + bias, name=scope)

        with tf.name_scope('conv4_3') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            self.params += [kernel, bias]
            self.conv4_3 = tf.nn.relu(self.con2d(self.conv4_2, kernel) + bias, name=scope)

        self.maxpool4 = tf.nn.max_pool(self.conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool4')

        # conv5
        with tf.name_scope('conv5_1') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            self.params += [kernel, bias]
            self.conv5_1 = tf.nn.relu(self.con2d(self.maxpool4, kernel) + bias, name=scope)

        with tf.name_scope('conv5_2') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            self.params += [kernel, bias]
            self.conv5_2 = tf.nn.relu(self.con2d(self.conv5_1, kernel) + bias, name=scope)

        with tf.name_scope('conv5_3') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            self.params += [kernel, bias]
            self.conv5_3 = tf.nn.relu(self.con2d(self.conv5_2, kernel) + bias, name=scope)

        self.maxpool5 = tf.nn.max_pool(self.conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool5')

        #fc6
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.maxpool5.get_shape()[1:]))
            kernel = self.getWeight([shape, 4096])
            bias = self.getBias([4096])
            self.params += [kernel, bias]
            pool5_flat = tf.reshape(self.maxpool5, [-1, shape])
            fc6_dropout = tf.nn.dropout(pool5_flat, 0.5)
            self.fc6 = tf.nn.relu(tf.matmul(fc6_dropout, kernel)+bias, name=scope)

        #fc7
        with tf.name_scope('fc7') as scope:
            kernel = self.getWeight([4096, 4096])
            bias = self.getBias([4096])
            self.params += [kernel, bias]
            fc7_dropout = tf.nn.dropout(self.fc6, 0.5)
            self.fc7 = tf.nn.relu(tf.matmul(fc7_dropout, kernel)+bias, name=scope)

        #fc8
        with tf.name_scope('fc8') as scope:
            kernel = self.getWeight([4096, self.classes])
            bias = self.getBias([self.classes])
            self.params += [kernel, bias]
            self.fc8 = tf.nn.relu(tf.matmul(self.fc7, kernel)+bias, name=scope)

        self.finaloutput = tf.nn.softmax(self.fc8, name="softmax")
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.finaloutput, labels=self.y))
        #self.optimizer = tf.train.MomentumOptimizer(self.learningRate, self.momentum, name='optimizer').minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.loss)
        self.prediction = tf.argmax(self.finaloutput, axis=1, name='predictions')

        return



    def con2d(self, x, kernel):
        return tf.nn.conv2d(x, kernel, [1,1,1,1],padding='SAME')

    def getWeight(self, shape, name='weight'):
        init = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(init, name=name, trainable=True)

    def getBias(self, shape, name='bias'):
        init = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(init, name=name, trainable=True)
