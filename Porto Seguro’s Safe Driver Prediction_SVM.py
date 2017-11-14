import tensorflow as tf
import numpy as np

'''
SVM, similar to Logistic Regression, is a method used to solve categorizing problem by
mapping previously linear inseparable points to a higher dimension space to achieve separability. 
It is widely used on binary prediction problems.
'''

#larger dataset help to converge
batch_size = 512
file_path = '/home/shr/software/machine_learning/kaggle/train.csv'
trained_result_save_path = '/home/shr/software/machine_learning/kaggle/'
x = []
y = []
file = open(file_path)
for line in file:
    #for each line, eliminate '\n' and split the line by ','
    x.append(line.strip('\n').split(',')[2:])
    #do the same for y
    y.append(1 if line.split(',')[1:2] == 1 else -1)
print (np.shape(x))
print (np.shape(y))

#skip header line
x = np.array(x[1:])
y = np.array(y[1:])

#fetch randomly chosen indices
train_indices = np.random.choice(len(x), int(len(x)*0.8), replace=False)
print (np.shape(train_indices))

test_indices = np.array(list(set(range(len(x))) - set(train_indices)))
x_vals_train = x[train_indices]
x_vals_test = x[test_indices]
y_vals_train = y[train_indices]
y_vals_test = y[test_indices]

#placeholders for input
x_data = tf.placeholder(shape=[None, 57], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[57, 1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1, 1]), dtype=tf.float32)

model_output = tf.subtract(tf.matmul(x_data, A), b)
l2_norm = tf.reduce_sum(tf.square(A))
#define alpha
alpha = tf.constant([0.1])

#hinge loss
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1.,tf.multiply(model_output, y_target))))
#hinge loss with regularization term
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    loss_vec = []
    train_accuracy = []
    test_accuracy = []
    for i in range(2000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train,
                                                       y_target: np.transpose([y_vals_train])})
        train_accuracy.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test,
                                                      y_target: np.transpose([y_vals_test])})
        test_accuracy.append(test_acc_temp)

        if (i + 1) % 10 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + 'b = ' + str(sess.run(b)))
            print('Loss = ' + str(temp_loss))

    result_A = sess.run(A)
    result_b = sess.run(b)
    np.savetxt(trained_result_save_path+"svm_A.txt", result_A)
    np.savetxt(trained_result_save_path+"svm_b.txt", result_b)


