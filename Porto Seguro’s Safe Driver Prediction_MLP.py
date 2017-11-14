import tensorflow as tf
import numpy as np

path = '/home/shr/software/machine_learning/kaggle/train.csv'
logs_path = '/home/shr/software/machine_learning/kaggle/'

# path = '/home/shr/software/greentea/1110greentea/imu.csv'
filename_queue = tf.train.string_input_producer([path])
num_epoch = 1
learning_rate = 0.001

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

record_defaults = [[1]]+[[1]]+[[0.1] for i in range(57)]

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, \
col15, co16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, \
co27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38, \
col39, col40, col41, col42, col43, col44, col45, col46, col47, col48, col49, col50, \
col51, col52, col53, col54, col55, col56, col57, col58, col59 = \
    tf.decode_csv(value, record_defaults=record_defaults)

feature = tf.stack([col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14,
                     col15, co16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26,
                     co27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38,
                     col39, col40, col41, col42, col43, col44, col45, col46, col47, col48, col49,
                     col50, col51, col52, col53, col54, col55, col56, col57, col58, col59])

# record_defaults = [[0.1] for i in range(59)]
#
# col1, col2, col3, col4, col5, col6, col7 = tf.decode_csv(value, record_defaults=record_defaults)

# features = tf.stack([col2, col3, col4, col5, col6])
#
label = [tf.one_hot(col2, 2)]

feature_batch, label_batch = tf.train.shuffle_batch(
    [feature, label], batch_size=256, capacity=640,
    min_after_dequeue=320)

X = tf.placeholder("float", [None, 57])
Y = tf.placeholder("float", [None, 1, 2])

print ("Shape of feature batch is : {}".format(feature_batch.shape))
print ("Shape of label batch is : {}".format(label_batch.shape))

def MLP(x):
    # input = x["image"]
    l1 = tf.layers.dense(x, 128)
    l2 = tf.layers.dense(l1, 256)
    l3 = tf.layers.dense(l2, 128)
    l4 = tf.layers.dense(l3, 64)
    l5 = tf.layers.dense(l4, 32)
    output = tf.layers.dense(l5, 2)
    return output

logits = MLP(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
epoch_num = 10
batch_num = 100
csv_size = sum(1 for row in open(path))

tf.summary.scalar("loss", loss_op)
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for _ in range(epoch_num):
        for i in range(int(csv_size/batch_num)):
            feature, label = sess.run([feature_batch, label_batch])
            #feed batch
            _, c = sess.run([train_op, loss_op], feed_dict={X: feature,
                                                            Y: label})
            print (c)
    coord.request_stop()
    coord.join(threads)

    print ("Training Finished!")
    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")


# def model_fn(features, labels, mode):
#     logits = NN(features)
#
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode, predictions= logits)
#     cost = tf.reduce_mean(-tf.reduce_sum(labels*tf.log(logits), reduction_indices=1))
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#     acc_op = tf.metrics.accuracy(labels=labels, predictions=logits)
#
#     estim_specs = tf.estimator.EstimatorSpec(
#         mode=mode,
#         predictions=logits,
#         loss=cost,
#         train_op=optimizer,
#         eval_metric_ops={'accuracy': acc_op})
#     return estim_specs
#
# print ("Define Model")
# model = tf.estimator.Estimator(model_fn)
#
# print ("Define Input Function")
# # Define the input function for training
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={input: features_batch}, y=label_batch,
#     batch_size=32, num_epochs=None, shuffle=True)
# # Train the Model
# print ("Start Training")
# model.train(input_fn, steps=100)
# print ("finish training")
#
# # Evaluate the Model
# # Define the input function for evaluating
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x=features, y=label,
#     batch_size=32, num_epochs=None, shuffle=True)
# # Use the Estimator 'evaluate' method
# e = model.evaluate(input_fn)
#
# print("Testing Accuracy:", e['accuracy'])