import tflearn as tfl
import tensorflow as tf


def weight_init(shape):
    weights = tf.truncated_normal(shape,mean=0, stddev=0.1, dtype=tf.float32)
    return tf.Variable(weights)

def biases_init(shape):
    biases = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(biases)


def mult(feature_map, row, height , times, flag=1):
    result = []
    temp_features = []
    for i in range(len(feature_map)):
        for j in range(times):
            W = weight_init([row, height,1])
            b = biases_init([1])
            temp_map = tf.multiply(feature_map[i], W) + b
            result.append(temp_map)
            temp_features.append(temp_map)
    if flag == 1:
        features_real.append(temp_features)
    else:
        features_imag.append(temp_features)
    return result

def _to_General(feature_map):
    result = feature_map[0]
    for i in range(1,len(feature_map)):
        result = tf.concat([result,feature_map[i]],3)
    return result

def win_size():
    return 32

features_real = []
features_imag = []



def mul_real(img):
    # ------------------param---------------------------------------
    size3 = win_size()
    temp = (128 - win_size()) // 2
    img3 = tf.slice(img, [0, temp, temp, 0], [-1, size3, size3, 1])
    feature_map3 = []
    feature_map3.append(img3)

    fea_map1 = 40

    # ------------------params---------------------------------------

    # -----------------------------------mult-----------------------------------
    feature_map3 = mult(feature_map3, size3, size3, fea_map1,flag=1)
    feature_map3 = mult(feature_map3, size3, size3, 1,flag=1)
    feature_map3 = mult(feature_map3, size3, size3, 1,flag=1)
    feature_map3 = mult(feature_map3, size3, size3, 1,flag=1)
    feature_map3 = mult(feature_map3, size3, size3, 1,flag=1)

    fc_net = _to_General(feature_map3)
    fc_net = tfl.max_pool_2d(fc_net, 2, 2)

    # -----------------------------------mult-----------------------------------
    return fc_net

def mul_imag(img):
    # ------------------param---------------------------------------
    size3 = win_size()
    temp = (128 - win_size()) // 2
    img3 = tf.slice(img, [0, temp, temp, 0], [-1, size3, size3, 1])
    feature_map3 = []
    feature_map3.append(img3)

    fea_map1 = 40


    # ------------------params---------------------------------------

    # -----------------------------------mult-----------------------------------
    feature_map3 = mult(feature_map3, size3, size3, fea_map1,flag=0)
    feature_map3 = mult(feature_map3, size3, size3, 1,flag=0)
    feature_map3 = mult(feature_map3, size3, size3, 1,flag=0)
    feature_map3 = mult(feature_map3, size3, size3, 1,flag=0)
    feature_map3 = mult(feature_map3, size3, size3, 1,flag=0)

    fc_net = _to_General(feature_map3)
    fc_net = tfl.conv_2d(fc_net, fea_map1, 5, activation='relu')
    fc_net = tfl.max_pool_2d(fc_net, 2, 2)
    # -----------------------------------mult-----------------------------------
    return fc_net

def MNN_1(fea1,fea2,drop=0.3):

    fc_net = tf.add(fea1,fea2)

    # -----------------------------------classification-----------------------------------

    fc_net = tfl.fully_connected(fc_net, 2048, activation='tanh', name='fc3')
    fc_net = tfl.dropout(fc_net, drop, name='drop1')

    fc_net = tfl.fully_connected(fc_net, 512, activation='tanh', name='fc5')
    fc_net = tfl.dropout(fc_net, drop, name='drop2')
    softmax = tfl.fully_connected(fc_net, 6, activation='softmax', name='prob')

    # -----------------------------------classification-----------------------------------

    return softmax

