#coding=utf-8
import cifar10,cifar10_input
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 4500
batch_size = 128

data_dir = './cifar10_data/cifar-10-batches-bin'
#下载好的数据集所在的文件夹

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        ## wl 控制l2损失的比重
        tf.add_to_collection('losses', weight_loss) 
        ## 参数的l2损失加入到损失集合中
    return var


def loss(logits, labels):

    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    # logits为[batch_size，num_classes]
    # labels为[batch_size，]的一维向量，其中每一个元素代表对应样本的类别
    # 先对网络的输出 Logits 进行 Softmax 概率化
    # Cross-Entropy 每个样本的交叉熵（损失）
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # 一个 batch 内样本求平均损失
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss') 
    # get 损失集合中所有损失，并相加后返回损失总和
  

# cifar10.maybe_download_and_extract()
# 如果没有下载，则需要将该行注释取消，
# 当然检查到 data_dir 目录下已经下载好的，则自动取消下载

images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                            batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)                                                  
#images_train, labels_train = cifar10.distorted_inputs()
#images_test, labels_test = cifar10.inputs(eval_data=True)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])
#原始图像是32*32*3，distorted_inputs函数随机裁剪，旋转成24*24*3的尺寸
#inputs用于测试集，只在正中间裁剪成24*24*3的尺寸
#logits = inference(image_holder)

#############################第一层卷积层###################################
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1)) #24*24*64
#####################################################################

#############################第二层池化与正则###############################
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')
#12*12*64
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75) 
#局部响应归一化，现在主要使用batch normalization
######################################################################

############################第三层卷积层#######################################
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))# 12*12*64
###########################################################################

#############################第四层池化与正则###############################
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')
#6*6*64
###########################################################################

reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value # 2304 = 6*6*64

############################第五层全连接##################################
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
#########################################################################


############################第六层全连接####################################
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))                                      
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
############################################################################

#############################第七层全连接###################################
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)
############################################################################


loss = loss(logits, label_holder) # 
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners()

## 实时显示
fig = plt.figure()
ax0 = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2)
plt.ion()
ax0.set_xlim(0,150)
ax0.set_ylim(0,1)

ax1.set_xlim(0,150)
ax1.set_ylim(0,2)

train_loss_all = [] 
test_loss_all =  []
train_precision_all = [] 
test_precision_all =  []

for e in range(max_steps):

    image_batch,label_batch = sess.run([images_train,labels_train])
    predictions, loss_train, _ = sess.run([top_k_op, loss, train_op],feed_dict={image_holder: image_batch, label_holder:label_batch})
    # run train_op 训练模型
    if e % 30 == 0:
        predictions_train = np.sum(predictions) / batch_size
        num_examples = 10000
        num_iter = int(math.ceil(num_examples / batch_size))
        true_count = 0  
        total_sample_count = num_iter * batch_size
        step = 0
        loss_value_list = []
        
        ##测试过程
        while step < num_iter:
    	    image_batch,label_batch = sess.run([images_test,labels_test])
    	    predictions,loss_value = sess.run([top_k_op,loss],feed_dict={image_holder: image_batch,
                                                 label_holder:label_batch})
    	    true_count += np.sum(predictions)
    	    loss_value_list.append(loss_value)
    	    step += 1

        precision_test = true_count / total_sample_count
        loss_test = sum(loss_value_list) / len(loss_value_list)
        print(str(e) + ",train_precision: " + str(predictions_train)[:5] +  ",train_loss: " + str(loss_train)[:5]  + "||" + "test_precision: " +  str(precision_test)[:5] + ",test_loss: " + str(loss_test)[:5])
        
        train_loss_all.append(loss_train)
        test_loss_all.append(loss_test)
        test_precision_all.append(precision_test)
        try:
            ax0.lines.remove(ax0_lines[0],ax0_lines[1])
            ax1.lines.remove(ax1_lines[0],ax1_lines[1])
        except Exception:
            pass 
        ax0_lines = ax0.plot(range(len(train_precision_all)), train_precision_all, c = 'b',label = "train_precision")
        ax0_lines = ax0.plot(range(len(test_precision_all)), test_precision_all, c = 'r',label = "test_precision")
        ax1_lines = ax1.plot(range(len(train_loss_all)), train_loss_all, c = 'b',label = "train_loss")
        ax1_lines = ax1.plot(range(len(test_loss_all)), test_loss_all, c = 'r',label = "test_loss")
        plt.pause(1)
        
    
plt.savefig("./fig.png")
        
        
        
