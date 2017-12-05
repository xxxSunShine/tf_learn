import os
import tensorflow as tf
import pprint
import numpy as np
from random import shuffle
from random import choice

image_width = 256
image_height =64

n_output_layer =10
keep_prob = tf.placeholder(tf.float32)

#车牌数据放在test_iamge目录下，根据具体目录修改
input_data_path = "./test_image"

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
area = ['京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕',
        '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

char_set = number + area + ALPHABET
CHAR_SET_LEN = len(char_set)

char_num_map = dict(zip(char_set,range(CHAR_SET_LEN)))

pprint.pprint(char_num_map)

#车牌长度为7
MAX_PLATENUMBER = 7

input_x = tf.placeholder(tf.float32,[None,image_height,image_width,3])
input_y = tf.placeholder(tf.float32,[None,CHAR_SET_LEN*MAX_PLATENUMBER])

input_datas =[]

def get_input_data(path):
    input_data = []

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(r'jpg'):
                tmp = []
                text = filename.strip(r'.jpg')
                file_path = os.path.join(path,filename)
                tmp.append(text)
                tmp.append(file_path)
                input_data.append(tmp)
    return input_data

input_datas = get_input_data(input_data_path)

shuffle(input_datas)

# 读取缩放图像
jpg_data = tf.placeholder(dtype=tf.string)
decode_jpg = tf.image.decode_jpeg(jpg_data, channels=3)
resize = tf.image.resize_images(decode_jpg, [image_height, image_width])
resize = tf.cast(resize, tf.uint8) / 255
def resize_image(file_name):
    with tf.gfile.FastGFile(file_name, 'r') as f:
        image_data = f.read()
    with tf.Session() as sess:
        image = sess.run(resize, feed_dict={jpg_data: image_data})
    return image

#将字符串转换为对应的向量
def text2vec(text):
    vector = np.zeros(MAX_PLATENUMBER * CHAR_SET_LEN)
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char_num_map.get(c)
        vector[idx] = 1
    return vector

pointer = 0
def get_train_batch(batch):
    global pointer
    batch_x = np.zeros([batch,image_height*image_width])
    batch_y = np.zeros([batch,CHAR_SET_LEN*MAX_PLATENUMBER])
    for i in range(batch):
        batch_x[i,:] = resize_image(input_datas[pointer][1])
        batch_y[i,:] = text2vec(input_datas[pointer][0])

    return batch_x,batch_y


def get_test_batch():
    batch_x = np.zeros([1000,image_height*image_width])
    batch_y = np.zeros([1000,CHAR_SET_LEN*MAX_PLATENUMBER])
    for i in range(1000):
        tem = choice[input_datas]
        batch_x[i,:] = resize_image(tem[1])
        batch_y[i,:] = text2vec(tem[0])
    return batch_x, batch_y



#定义卷积核
w_c1 = tf.Variable(0.01 * tf.random_normal([3, 3, 3, 32]))
b_c1 = tf.Variable(0.1 * tf.random_normal([32]))

w_c2 = tf.Variable(0.01 * tf.random_normal([3, 3, 32, 64]))
b_c2 = tf.Variable(0.1 * tf.random_normal([64]))

w_c3 = tf.Variable(0.01 * tf.random_normal([3, 3, 64, 64]))
b_c3 = tf.Variable(0x1 * tf.random_normal([64]))

w_d = tf.Variable(0.01 * tf.random_normal([8 * 64 * 32, 1024]))
b_d = tf.Variable(0.1 * tf.random_normal([1024]))


w_out = tf.Variable(0.01 * tf.random_normal([1024, CHAR_SET_LEN*MAX_PLATENUMBER]))
b_out = tf.Variable(0.1 * tf.random_normal([CHAR_SET_LEN*MAX_PLATENUMBER]))


def cnn_net(X):
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 全连接层
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


batch_size =64



def training():
    output = cnn_net(input_x)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=input_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    epochs = 100
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        saver = tf.train.Saver(tf.all_variables())
        for epoch in range(epochs):
            for i in range( int(len(input_datas)/batch_size) ):
                x, y = get_train_batch(batch_size)
                _, c = session.run([optimizer, loss], feed_dict={input_x: x, input_y:y,keep_prob:1})
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)
            if epoch % 10 == 0:
                saver.save(session, './platenumber.module', global_step=epoch)

        correct = tf.equal(tf.argmax(output,1), tf.argmax(input_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        x, y = get_test_batch(batch_size)
        print('准确率: ', accuracy.eval({input_x:x, input_y:y,keep_prob:1}))


