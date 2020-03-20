#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:49:06 2019

@author: nickwang
"""

import tensorflow as tf
import numpy as np
import time, os
from DataLoader_ILSVRC import ILSVRC2012
import matplotlib.pyplot as plt


def convLayer(inputs, kernel_shape, bias_init=1, padding="SAME", stride=1):
    weights = tf.Variable(np.random.normal(scale=0.01, size=kernel_shape), dtype=tf.float32)
    biases = tf.Variable(tf.constant(bias_init, shape=[kernel_shape[-1]], dtype=tf.float32))
#    initializer = tf.contrib.layers.xavier_initializer()
#    weights = tf.Variable(initializer(kernel_shape))
#    biases = tf.Variable(initializer([kernel_shape[-1]]))
    
    outputs = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1], padding=padding)
#    outputs = tf.layers.batch_normalization(outputs)
    outputs = tf.nn.relu(tf.nn.bias_add(outputs, biases))    
    return outputs

def LRNLayer(inputs):
    return tf.nn.local_response_normalization(inputs, depth_radius=2, bias=2, alpha=1e-4, beta=0.75)

def PoolingLayer(inputs):
    return tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

def FullyLayer(inputs, kernel_shape, activation='RELU'):
    weights = tf.Variable(np.random.normal(scale=0.01, size=kernel_shape), dtype=tf.float32)
    biases = tf.Variable(tf.constant(1, shape=[kernel_shape[-1]], dtype=tf.float32))    
#    initializer = tf.contrib.layers.xavier_initializer()
#    weights = tf.Variable(initializer(kernel_shape))
#    biases = tf.Variable(initializer([kernel_shape[-1]]))  
    
    outputs = tf.add(tf.matmul(inputs, weights), biases)
    if activation=='RELU':
#        outputs = tf.layers.batch_normalization(outputs)
        outputs = tf.nn.relu(outputs)
    return outputs

def DropOutLayer(inputs, keep_prob=0.5):
    return tf.nn.dropout(inputs, keep_prob=keep_prob)

def OutputLayer(y_pred, y_true):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

class AlexNet(object):
    def __init__(self, keep_prob, num_classes):
        self.keep_prob = keep_prob
        self.num_classes = num_classes 

    def inference(self, inputs):
        C1 = convLayer(inputs, [11, 11, 3, 96], bias_init=0, padding="VALID", stride=4)
        N1 = LRNLayer(C1)
        P1 = PoolingLayer(N1)
        C2_1 = convLayer(P1[..., :48], [5, 5, 48, 128])
        C2_2 = convLayer(P1[..., 48:], [5, 5, 48, 128])
        N2 = LRNLayer(tf.concat([C2_1, C2_2], axis=-1))
        P2 = PoolingLayer(N2)
        C3 = convLayer(P2, [3, 3, 256, 384], bias_init=0)
        C4_1 = convLayer(C3[..., :192], [3, 3, 192, 192])
        C4_2 = convLayer(C3[..., 192:], [3, 3, 192, 192])
        C5_1 = convLayer(C4_1, [3, 3, 192, 128])
        C5_2 = convLayer(C4_2, [3, 3, 192, 128])
        P5 = PoolingLayer(tf.concat([C5_1, C5_2], axis=-1))
        F6 = FullyLayer(tf.layers.flatten(P5), [9216, 4096])
        D6 = DropOutLayer(F6, keep_prob=self.keep_prob)
        F7 = FullyLayer(tf.layers.flatten(D6), [4096, 4096])
        D7 = DropOutLayer(F7, keep_prob=self.keep_prob)
        F8 = FullyLayer(tf.layers.flatten(D7), [4096, self.num_classes], activation='')
        
        return F8
        
if __name__ == "__main__":
    
    restore_path = None
    
    trainingSet = ILSVRC2012('/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_train', 'dirname_to_classname')
    testingSet = ILSVRC2012('/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_val', 'dirname_to_classname')
    
    num_classes = trainingSet.num_classes
    batch_size_train = 128
    batch_size_test = 256
#    momentum = 0.9  # When I use MomentumOpt, it stuck at a saddle point, so I replace with Adam.
    learning_rate = [0.01] * 30 + [0.001] * 25 + [0.0001] * 20 + [0.00001] * 15
    learning_rate = [lr*0.01 for lr in learning_rate] # The original lr is too big to find the global minimum loss.
        
    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y_true = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)  
    learning_holder = tf.placeholder(tf.float32)
    alexnet = AlexNet(keep_prob, num_classes)    
    y_pred = alexnet.inference(x)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))
#    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_holder, momentum=momentum).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_holder).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if restore_path != None:
            saver.restore(sess, restore_path)
        else:
            sess.run(tf.global_variables_initializer())
    
        train_loss_list = list()
        train_accuracy_list = list()
        test_loss_list = list()
        test_accuracy_list = list()
        train_data_index = np.arange(len(trainingSet))    
        
        
        for epoch in range(len(learning_rate)):
            print("=================================================")
            print("epoch \t", epoch + 1, "\n")
            np.random.shuffle(train_data_index)
            
            
            time_start = time.time()
            for i in range(len(trainingSet) // batch_size_train):
                train_imgs, train_labels = trainingSet.__getitem__(train_data_index[i*batch_size_train:(i+1)*batch_size_train])
                train_labels_onehot = np.zeros((batch_size_train, num_classes))
                train_labels_onehot[range(len(train_labels)), train_labels] = 1
                _, y_pred_print, loss_print = sess.run([optimizer, y_pred,  loss], feed_dict={x: train_imgs, y_true: train_labels_onehot, keep_prob: 0.5, learning_holder: learning_rate[epoch]})            
                
                time_end = time.time()
                   
                if ((i + 1) % 1) == 0:
                    print("Training Data Num ", (i + 1) * batch_size_train, "Loss = ", loss_print, 'Batch Accuracy : %.2f%%' %(np.sum(np.equal(np.argmax(train_labels_onehot, axis=1), np.argmax(y_pred_print, axis=1)))/batch_size_train*100))
                    print("labels(GT) = ", train_labels[:10])
                    print("labels(PD) = ", np.argmax(y_pred_print[:10], axis=1))


            acc_train = 0
            loss_train = 0
            for i in range(len(trainingSet) // batch_size_train):
                train_imgs, train_labels = trainingSet.__getitem__(range(i*batch_size_train, (i+1)*batch_size_train))
                train_labels_onehot = np.zeros((batch_size_train, num_classes))
                train_labels_onehot[range(len(train_labels)), train_labels] = 1
                y_pred_print, loss_print = sess.run([y_pred,  loss], feed_dict={x: train_imgs, y_true: train_labels_onehot, keep_prob:1}) 
                acc_train += np.sum(np.equal(train_labels, np.argmax(y_pred_print, axis=1)))
                loss_train += loss_print
            acc_train /= ((len(trainingSet) // batch_size_train) * batch_size_train)
            loss_train /= len(trainingSet) // batch_size_train
            train_loss_list.append(loss_train)
            train_accuracy_list.append(acc_train)
            print("Train Loss : ", loss_train, "Accuracy : %.2f%%" %(acc_train * 100))
            
            acc_test = 0
            loss_test = 0
            for i in range(len(testingSet) // batch_size_test):
                test_imgs, test_labels = testingSet.__getitem__(range(i*batch_size_test, (i+1)*batch_size_test))
                test_labels_onehot = np.zeros((batch_size_test, num_classes))
                test_labels_onehot[range(len(test_labels)), test_labels] = 1
                y_pred_print, loss_print = sess.run([y_pred,  loss], feed_dict={x: test_imgs, y_true: test_labels_onehot, keep_prob: 1})   
                loss_test += loss_print
                acc_test += np.sum(np.equal(np.argmax(test_labels_onehot, axis=1), np.argmax(y_pred_print, axis=1)))
            acc_test /= ((len(testingSet) // batch_size_test) * batch_size_test)
            loss_test /= len(testingSet) // batch_size_test
            test_loss_list.append(loss_test)
            test_accuracy_list.append(acc_test)
            print("Test Loss : ", loss_test, "Accuracy : %.2f%%" %(acc_test * 100))
            
            if not os.path.isdir("./weights/"):
                os.mkdir("./weights/")
            save_path = saver.save(sess, "./weights/AlexNetWeights_epoch_" +str(epoch + 1) + '_iter_' + str((i + 1) * batch_size_train) + ".ckpt")

    x = np.arange(len(learning_rate)) + 1
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.plot(x, train_accuracy_list)
    plt.plot(x, test_accuracy_list)
    plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
    plt.show()
    plt.savefig('Accuracy.png') 
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.plot(x, train_loss_list)
    plt.plot(x, test_loss_list)
    plt.legend(['training loss', 'testing loss'], loc='upper right')
    plt.show()
    plt.savefig('Loss.png') 