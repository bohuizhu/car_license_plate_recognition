# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:12:33 2018

@author: 83400
"""

#!/usr/bin/python3.5

# -*- coding: utf-8 -*-  

 

import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
from PIL import Image
size = 1280

WIDTH = 32

HEIGHT = 40

NUM_CLASSES = 34


 

SAVER_DIR = "train-saver/digits/"

 

LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z")

license_num = ""

 

x = tf.placeholder(tf.float32, shape=[None, size])

y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

 
# define image
x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])



if __name__ =='__main__' and sys.argv[1]=='train':

    # the total number of pics for train dataset
    
    input_count = 0

    for i in range(0,NUM_CLASSES):

        dir = './train_images/training-set/%s/' % i  

        for _, _, files in os.walk(dir):

            for filename in files:

                input_count += 1

 

    # 定义对应维数和各维长度的数组

    input_images = np.array([[0]*size for i in range(input_count)])

    input_labels = np.array([[0]*NUM_CLASSES for i in range(input_count)])

 

    # get pic and label

    index = 0

    for i in range(0,NUM_CLASSES):

        
        dir = './train_images/training-set/%s/' % i            

        for _, _, files in os.walk(dir):

            for filename in files:

                filename = dir + filename

                img = Image.open(filename)

                width = img.size[0]

                height = img.size[1]

                for h in range(0, height):

                    for w in range(0, width):

                        # make pic thinner, increase accuracy 

                        if img.getpixel((w, h)) > 230:

                            input_images[index][w+h*width] = 0

                        else:

                            input_images[index][w+h*width] = 1

                input_labels[index][i] = 1

                index += 1

 

    # total pic and label for validation dataset

    val_count = 0

    for i in range(0,NUM_CLASSES):

        dir = './train_images/validation-set/%s/' % i   

        for _, _, files in os.walk(dir):

            for filename in files:

                val_count += 1

 

    # 定义对应维数和各维长度的数组

    val_images = np.array([[0]*size for i in range(val_count)])

    val_labels = np.array([[0]*NUM_CLASSES for i in range(val_count)])

 

    # 第二次遍历图片目录是为了生成图片数据和标签

    index = 0

    for i in range(0,NUM_CLASSES):

        dir = './train_images/validation-set/%s/' % i   

        for _, _, files in os.walk(dir):

            for filename in files:

                filename = dir + filename

                img = Image.open(filename)

                width = img.size[0]

                height = img.size[1]

                for h in range(0, height):

                    for w in range(0, width):

                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率

                        if img.getpixel((w, h)) > 230:

                            val_images[index][w+h*width] = 0

                        else:

                            val_images[index][w+h*width] = 1

                val_labels[index][i] = 1

                index += 1

    

    with tf.Session() as sess:

        # 1st cnn layer with pooling 

        W_c1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_c1")

        b_c1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_c1")

        conv_strides = [1, 1, 1, 1]
        L1_conv = tf.nn.conv2d(x_image, W_c1, strides=conv_strides, padding='SAME')
        L1_relu = tf.nn.relu(L1_conv + b_c1)

        pool_strides = [1, 2, 2, 1]
        kernel_size = [1, 2, 2, 1]
        L1_pool = tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')

 

        # 2ed cnn layer with pooling 

        W_c2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name="W_c2")

        b_c2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_c2")

        conv_strides = [1, 1, 1, 1]
        L2_conv = tf.nn.conv2d(L1_pool, W_c2, strides=conv_strides, padding='SAME')
        L2_relu = tf.nn.relu(L2_conv + b_c2)

        kernel_size = [1, 1, 1, 1]

        pool_strides = [1, 1, 1, 1]

        L2_pool = tf.nn.max_pool(L2_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')

        #fully-connected layer

        W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")

        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")

        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20*32])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        # dropout

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout

        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name="W_fc2")

        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")


        # training 

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)

 

        correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

 

        sess.run(tf.global_variables_initializer())
        
        print ("total get %s,training pics,%stags" % (input_count, input_count))


        batch_size = 60

        iterations = 30

        batches_count = int(input_count / batch_size)

        remainder = input_count % batch_size

        print ("trainig set is divided into %s,each set is: %s,the last set is: %s" % (batches_count+1, batch_size, remainder))

 

        # train iteration  dropout:50%

        for it in range(iterations):

            for n in range(batches_count):

                train_step.run(feed_dict={x: input_images[n*batch_size:(n+1)*batch_size], y_: input_labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})

            if remainder > 0:

                start_index = batches_count * batch_size;

                train_step.run(feed_dict={x: input_images[start_index:input_count-1], y_: input_labels[start_index:input_count-1], keep_prob: 0.5})


            iterate_accuracy = 0

            if it%5 == 0:

                iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})

                print ('%dth: accurate rate is :%0.5f%% ' % (it+1, iterate_accuracy*100))

                if iterate_accuracy >= 0.9999 and it >= iterations:

                    break;


        print ('finish traning!')
        
        # keep train ouyput
        if not os.path.exists(SAVER_DIR):

            print ('do not save file')

            os.makedirs(SAVER_DIR)

        # initialize trainer 

        saver = tf.train.Saver()            

        saver_path = saver.save(sess, "%smodel.ckpt"%(SAVER_DIR))

 


if __name__ =='__main__' and sys.argv[1]=='predict':

    saver = tf.train.import_meta_graph("%smodel.ckpt.meta"%(SAVER_DIR))

    with tf.Session() as sess:

        model_file=tf.train.latest_checkpoint(SAVER_DIR)

        saver.restore(sess, model_file)

       

        W_c1 = sess.graph.get_tensor_by_name("W_c1:0")

        b_c1 = sess.graph.get_tensor_by_name("b_c1:0")

        conv_strides = [1, 1, 1, 1]
        L1_conv = tf.nn.conv2d(x_image, W_c1, strides=conv_strides, padding='SAME')
        L1_relu = tf.nn.relu(L1_conv + b_c1)

        kernel_size = [1, 2, 2, 1]

        pool_strides = [1, 2, 2, 1]

        L1_pool = tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')
 

       

        W_c2 = sess.graph.get_tensor_by_name("W_c2:0")

        b_c2 = sess.graph.get_tensor_by_name("b_c2:0")

        conv_strides = [1, 1, 1, 1]
        L2_conv = tf.nn.conv2d(L1_pool, W_c2, strides=conv_strides, padding='SAME')
        L2_relu = tf.nn.relu(L2_conv + b_c2)

        kernel_size = [1, 1, 1, 1]

        pool_strides = [1, 1, 1, 1]
        L2_pool = tf.nn.max_pool(L2_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')

 

 

       # fully connected layer 

        W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")

        b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")

        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

 

 

        # dropout

        keep_prob = tf.placeholder(tf.float32)

 

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

 

 

        # readout

        W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")

        b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")

 

        # training 

        conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

 

        for n in range(0,4):

            path = r"test_images/%s.bmp"  %n

            img = Image.open(path)

            width = img.size[0]

            height = img.size[1]


            img_data = [[0]*size for i in range(1)]

            for h in range(0, height):

                for w in range(0, width):

                    if img.getpixel((w, h)) < 190:

                        img_data[0][w+h*width] = 1

                    else:

                        img_data[0][w+h*width] = 0

            

            result = sess.run(conv, feed_dict = {x: np.array(img_data), keep_prob: 1.0})


            max1 = 0

            max2 = 0

            max3 = 0

            max1_index = 0

            max2_index = 0

            max3_index = 0

            for j in range(NUM_CLASSES):

                if result[0][j] > max1:

                    max1 = result[0][j]

                    max1_index = j

                    continue

                if (result[0][j]>max2) and (result[0][j]<=max1):

                    max2 = result[0][j]

                    max2_index = j

                    continue


            license_num = license_num + LETTERS_DIGITS[max1_index]

            print ("proability:  [%s %0.2f%%] [%s %0.2f%%] " % (LETTERS_DIGITS[max1_index],max1*100, LETTERS_DIGITS[max2_index],max2*100,))

            

        print ("car plate is [%s]" % license_num)