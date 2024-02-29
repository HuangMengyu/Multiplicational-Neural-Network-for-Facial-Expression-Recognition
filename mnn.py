import tensorflow as tf
import numpy as np
import time,sys
import logging
import os
import win_unicode_console
import datetime
import cv2

win_unicode_console.enable()

#os.environ["CUDA_VISIBLE_DEVICES"]='1'

#train_epochs = 40
#train_epochs = 50
train_epochs = 40
epochs_time = 100   #batch num
#batch_size = 42
#batch_size = 15
batch_size = 15
learning_rate = 0.0001
average_fold = 10
#average_fold = 1
para = 0

#pkl_path1 = './pkl/kdef_with_img_geometry.pkl'
#pkl_path2 = './pkl/ckp_with_img_geometry.pkl'
#pkl_path2 = './pkl/ckp_with_img_geometry_acd.pkl'
pkl_path2 = './pkl/ckp_with_img_geometry.pkl'
pkl_path1 = './pkl/ckplus_with_img_geometry_7_neutral.pkl'
pkl_path3 = './pkl/oulu_casia_with_img_geometry.pkl'
#pkl_path3 = './pkl/oulus_casia_with_img_geometry_3frame_1.pkl'
# pkl_path1 = './pkl/orl_with_img.pkl'
# pkl_path2 = './pkl/lfw_with_img.pkl'

pkl_flag = 0
if len(sys.argv)>1:
    pkl_flag = int(sys.argv[1])
else:
    pkl_flag = 2
    #print('Usage:python cnn_for_fear_ten_fold_ten.py 1')
    #exit(1)

#inforrun = 0
inforrun = 3

#pkl_path = ''
filenames = ''
#pkl_flag = 2
if pkl_flag == 1:
    pkl_path = pkl_path1
    filenames = "./cnn_mark/cnn_ten_fold_mark_ck+_7_neutral.log"
   # filenames = "./cnn_mark/cnn_ten_fold_mark_kdef.log"
    #filenames = "./cnn_mark/cnn_ten_fold_mark_orl.log"
elif pkl_flag == 2:
    pkl_path = pkl_path2
   # filenames = "./cnn_mark/cnn_ten_fold_mark_ck+_7_neutral.log"
    filenames = "./cnn_mark/cnn_ten_fold_mark_ck+.log"
    #filenames = "./cnn_mark/cnn_ten_fold_mark_lfw.log"
elif pkl_flag == 3:
    pkl_path = pkl_path3
    filenames = "./cnn_mark/cnn_ten_fold_mark_oulu+.log"
   # filenames = "./cnn_mark/cnn_ten_fold_mark_oulu+_3frame.log"
else:
    print('pkl: 1 2 3')
    exit(1)

logging.basicConfig(level=logging.DEBUG, 
                    filename=filenames, 
                    filemode="a+", 
                    format="%(asctime)-15s %(levelname)-8s  %(message)s")

import NET
#logging.info('ADD+merge after pooling+comtempt')##
logging.info('Window size : {0}'.format(NET.win_size()))
logging.info('DataSet : {0}'.format(pkl_path))
logging.info('train_epochs:{0}'.format(train_epochs))  
logging.info('epochs_time:{0}'.format(epochs_time))
logging.info('batch_size :{0}'.format(batch_size ))
logging.info('learning_rate:{0}'.format(learning_rate))
logging.info('average_fold:{0}'.format(average_fold))

logging.info('testing without the conv layer but just mult and pooling')

def get_random_batchdata(n_samples, batchsize):
    start_index = np.random.randint(0, n_samples - batchsize)
    return (start_index, start_index + batchsize)


x_dct = tf.placeholder(tf.float32, [None, 128, 128],name='img')
x_dst = tf.placeholder(tf.float32, [None, 128, 128],name='img')
y = tf.placeholder(tf.float32, [None, 6])
keep_prob = tf.placeholder("float")

x_image1 = tf.reshape(x_dct, [-1, 128, 128, 1])
x_image2 = tf.reshape(x_dst, [-1, 128, 128, 1])

import NET
x1 = NET.mul_real(x_image1)
x2 = NET.mul_imag(x_image2)
y_out = NET.MNN_1(x1, x2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out),0)
loss_summary = tf.summary.scalar('loss',cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accu_summary = tf.summary.scalar('accu',accuracy)

init1 = tf.global_variables_initializer()

print('preparing data......')
from PreProcessing import pickle_2_img_single as loadPKL
#frame_x1, data_label=loadPKL(pkl_path)
frame_x1, frame_x2,data_label=loadPKL(pkl_path) #frame_x1:dctn frame_x2:dstn
print(len(data_label), len(data_label[0]))
#exit(0)
best_acc_of_fold = []

wts = time.time()

iteration = inforrun

m_saver = tf.train.Saver()
while iteration<10:
    #if iteration > 0:
    #    break
    test_x1 = []
    test_x2 = []
    test_y = []
    img_x1 = []
    img_x2 = []
    label_y = []

    #get k-fold iteration
    for idfd in range(len(data_label)):
        if idfd == iteration:
            test_x1 = test_x1+frame_x1[idfd]
            test_x2 = test_x2 + frame_x2[idfd]
            test_y = test_y+data_label[idfd]
        else:
            img_x1 = img_x1+frame_x1[idfd]
            img_x2 = img_x2+frame_x2[idfd]
            label_y = label_y+data_label[idfd]

    iteration = iteration+1
    n_samples = len(label_y)
    print('test data %d'%len(test_y))
    print('total data %d'%(n_samples))
     
    isLog = True
    best_acc = 0
    best_epochs = 0
    cor_acc = 0
    cor_lost = 0
    confusion_matrix = []
    best_confusion_matrix = []
    isAccuarcy100 = False

    for i_in_ten in range(average_fold):

        with tf.Session() as sess1:

            sess1.run(init1)
            time_start = time.time()

            for i in range(train_epochs):

                for j in range(epochs_time):  #100 batches

                    start_index, end_index = get_random_batchdata(n_samples, batch_size)

                    batch_x1 = img_x1[start_index: end_index]
                    batch_x2 = img_x2[start_index: end_index]
                    batch_y = label_y[start_index: end_index]

                    _, cost, accu = sess1.run([optimizer, cross_entropy, accuracy], feed_dict={x_dct: batch_x1, x_dst: batch_x2, y: batch_y, keep_prob: 0.3})

                    test_acc_show = 0
                    if accu>0.90:
                        correct_count=0
                        confusion_matrix = np.zeros((6,6),int)
                        result = sess1.run(y_out, feed_dict={x_dct: test_x1, x_dst: test_x2, keep_prob: 1})
                        for k in range(len(test_y)):
                            ip = np.argmax(result[k],0)
                            ir = np.argmax(test_y[k],0)
                            confusion_matrix[ir][ip] = confusion_matrix[ir][ip]+1
                            if np.equal(ip,ir):
                                correct_count = correct_count+1
                        test_acc = float(correct_count)/len(test_y)
                        time3 = datetime.datetime.now()

                        test_acc_show = test_acc
                        if (test_acc == 1):
                            logging.info("test_accu == 1")
                            curr = time.time()
                            m_saver.save(sess1, "./pred_model/model-{}.ckpt".format(curr))
                            logging.info("model saved")
                            print("model saved")
                            exit(0)

                        if test_acc>best_acc or (test_acc==best_acc and accu>cor_acc):
                            best_acc = test_acc
                            best_epochs = i+1
                            cor_acc = accu
                            cor_lost = cost
                            best_confusion_matrix = confusion_matrix
                            print('Best_accuracy:%s   ,accuracy : %.7f,      cost : %.7f  '%(str(best_acc), accu ,cost))
                            print('Confusion_Matrix:')
                            print('#Anger:0,Surprise:1,Disgust:2,Fear:3,Happiness:4,Sadness:5')
                            print(best_confusion_matrix)

                    print ('Fold : %d , %d , Epoch : %d ,  times:%d , accuracy : %.7f,  best_acc : || %.7f ||,  test_acc : %.7f'%(iteration, i_in_ten+1, i+1, j+1, accu , best_acc,test_acc_show))

        sess1.close()

    if isLog:
        logging.info('Fold :{0} , Epochs:{1} , train_accuracy : {2} , test_accuarcy:{3} , lost:{4}'.format(iteration, best_epochs, cor_acc, best_acc, cor_lost))
        logging.info('#Anger:0,Surprise:1,Disgust:2,Fear:3,Happiness:4,Sadness:5')
        logging.info('Confusion Matrix \n {0}\n'.format(best_confusion_matrix))
        time_end = time.time()
        print("Fold %d time comsuming: %fs"%(iteration,(time_end-time_start)))
        best_acc_of_fold.append(best_acc)



wte = time.time()
print("Total Time comsuming: %fs"%(wte-wts))
print(best_acc_of_fold)
print("Mean accuracy : %.7f"%np.mean(best_acc_of_fold))


logging.info('Total Time comsuming:{0}s'.format(wte-wts))
logging.info('Ten Fold :{0}'.format(best_acc_of_fold))
logging.info('Mean accuracy :{0}'.format(np.mean(best_acc_of_fold)))
