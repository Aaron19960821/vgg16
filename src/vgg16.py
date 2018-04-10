#!coding=utf-8

import tensorflow as tf
import numpy as np
import json
import utils
import cv2
import time

class Vgg16:

    def __init__(self):
        self.loaded = False
        self.width = 228
        self.height = 228

    def loadWithUntrainedJson(self, jsonFile):
    # Todo: load train information with a JSON file
        self.trained = False
        data = utils.readJsonFromFile(jsonFile)

        self.learningRate = float(data['learning_rate'])
        self.momentum = float(data['momentum'])
        self.batchsize = int(data['batchsize'])
        self.batches = int(data['batches'])
        self.channel = int(data['channel'])
        self.classes = int(data['classes'])

        self.trainList = str(data['trainlist'])
        self.labelList = str(data['labelList'])
        

    def loadWithTrainedJson(self, jsonFile):
    # Todo: load train information with a JSON file
        self.trained = True

    def train(self, tgtDir):
        init = tf.global_variables_initializer()

        trainSet = open(self.trainList).readlines()
        initlabel = open(self.labelList).readlines()
        label = [int(x) for x in initlabel]

        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.batches):
                startTime = time.time()
                imageSet = utils.randomChoose(trainSet, self.batchsize)
                x = utils.loadImagesFromFile(imageSet, self.width, self.height)
                imageLabel = utils.getLabels(trainSet, imageSet, label)

                y = []
                for i in range self.batchsize:
                    y.append([1 if x==imageLabel[i]] else 0 for x in range(self.classes))
                sess.run(self.optimizer, feed_dict={
                    self.x: x,
                    self.y: np.array(y)
                    })
                endTime = time.time()

                print("Batch #{} processing time {}, loss={}", i+1, endTime-startTime, self.loss)

            self.saveAll()





        

    # Build the basic structure of network
    def buildNet(self):
        self.x = tf.placeholder(tf.float32, shape = (None, self.width, self.height, self.channel), name= 'input_layer')
        self.y = tf.placeholder(tf.int32, shape = (None, self.classes), name = 'labels')

        # Conv1
        with tf.name_scope('conv1_1') as scope:
            kernel = self.getWeight([3,3,3,64])
            bias = self.getBias([64])
            conv1_1 = tf.nn.relu(self.con2d(self.x, kernel)+bias, name = scope)

        with tf.name_scope('conv1_2') as scope:
            kernel = self.getWeight([3,3,64,64])
            bias = self.getBias([64])
            conv1_2 = tf.nn.relu(self.con2d(conv1_1, kernel)+bias, name = scope)

        maxpool1 = tf.nn.maxpool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool1')

        with tf.name_scope('conv2_1') as scope:
            kernel = self.getWeight([3,3,64,128])
            bias = self.getBias([128])
            conv2_1 = tf.nn.relu(self.con2d(maxpool1,kernel)+bias, name=scope)

        with tf.name_scope('conv2_2') as scope:
            kernel = self.getWeight([3,3,128,128])
            bias = self.getBias([128])
            conv2_2 = tf.nn.relu(self.con2d(conv2_1, kernel)+bias, name=scope)

        maxpool2 = tf.nn.maxpool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool2')

        # conv3
        with tf.name_scope('conv3_1') as scope:
            kernel = self.getWeight([3, 3, 128, 256])
            bias = self.getBias([256])
            conv3_1 = tf.nn.relu(self.con2d(maxpool2, kernel) + bias, name=scope)

        with tf.name_scope('conv3_2') as scope:
            kernel = self.getWeight([3, 3, 256, 256])
            bias = self.getBias([256])
            conv3_2 = tf.nn.relu(self.con2d(conv3_1, kernel) + bias, name=scope)

        with tf.name_scope('conv3_3') as scope:
            kernel = self.getWeight([3, 3, 256, 256])
            bias = self.getBias([256])
            conv3_3 = tf.nn.relu(self.con2d(conv3_2, kernel) + bias, name=scope)

        maxpool3 = tf.nn.maxpool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool3')

        # conv4
        with tf.name_scope('conv4_1') as scope:
            kernel = self.getWeight([3, 3, 256, 512])
            bias = self.getBias([512])
            conv4_1 = tf.nn.relu(self.con2d(maxpool3, kernel) + bias, name=scope)

        with tf.name_scope('conv4_2') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            conv4_2 = tf.nn.relu(self.con2d(conv4_1, kernel) + bias, name=scope)

        with tf.name_scope('conv4_3') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            conv4_3 = tf.nn.relu(self.con2d(conv4_2, kernel) + bias, name=scope)

        maxpool4 = tf.nn.maxpool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool4')


        # conv5
        with tf.name_scope('conv5_1') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            conv5_1 = tf.nn.relu(self.con2d(maxpool4, kernel) + bias, name=scope)

        with tf.name_scope('conv5_2') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            conv5_2 = tf.nn.relu(self.con2d(conv5_1, kernel) + bias, name=scope)

        with tf.name_scope('conv5_3') as scope:
            kernel = self.getWeight([3, 3, 512, 512])
            bias = self.getBias([512])
            conv5_3 = tf.nn.relu(self.con2d(output_conv5_2, kernel) + bias, name=scope)

        maxpool5 = tf.nn.maxpool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool5')

        #fc6
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            kernel = self.getWeight([shape, 4096])
            bias = self.getBias([4096])
            pool5_flat = tf.reshape(pool5, [-1, shape])
            output_fc6 = tf.nn.relu(tf.matmul(pool5_flat, kernel)+bias, name=scope)

        #fc7
        with tf.name_scope('fc7') as scope:
            kernel = self.getWeight([4096, 4096])
            bias = self.getBias([4096])
            output_fc7 = tf.nn.relu(tf.matmul(output_fc6, kernel)+bias, name=scope)

        #fc8
        with tf.name_scope('fc8') as scope:
            kernel = self.getWeight([4096, self.classes])
            bias = self.getBias([self.classes])
            output_fc8 = tf.nn.relu(tf.matmul(output_fc7, kernel)+bias, name=scope)

        finaloutput = tf.nn.softmax(output_fc8, name="softmax")
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=finaloutput, labels=y))

        self.optimizer = tf.train.MomentumOptimizer(self.learningRate, self.momentum, name='optimizer').minimize(self.loss)

        return



    def con2d(self, x, kernel):
        return tf.nn.conv2d(x, kernel, [1,1,1,1], padding='same')

    def getWeight(self, shape, name='weight'):
        init = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(init, name=name)

    def getBias(self, shape, name='bias'):
        init = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(init, name=name)
