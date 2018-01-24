
import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class Model(object):
    def __init__(self, data):
        #hyperparams
        learning_rate = 1e-4
        training_epochs = 1000
        batch_size = 200
        measure_frequency = 100
        beta = 0.01

        #model
        size_y = data.test.labels[0].size
        size_x = data.test.signals[0].size

        y = tf.placeholder(tf.float32, [None, size_y])
        x = tf.placeholder(tf.float32, [None, size_x])
        keep_prob = tf.placeholder(tf.float32)

        #dense
        hl_units = int((size_y + size_x)/2)
        dense = tf.layers.dense(inputs=x, units=hl_units, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=beta))
        dropout = tf.nn.dropout(dense, keep_prob)

        #readout
        W = weight_variable([hl_units, size_y])
        b = bias_variable([size_y])

        with tf.name_scope('Model'):
            model = tf.matmul(dropout, W) + b

        #optimisation
        with tf.name_scope('Loss'):
            loss =  tf.losses.softmax_cross_entropy(y , model) + tf.losses.get_regularization_loss() + beta * tf.nn.l2_loss(W)

        with tf.name_scope('SGD'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        #accuracy
        with tf.name_scope('Training_Accuracy'):
            train_accuracy = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
            train_accuracy = tf.reduce_mean(tf.cast(train_accuracy, tf.float32))

        with tf.name_scope('Validation_Accuracy'):
            valid_accuracy = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
            valid_accuracy = tf.reduce_mean(tf.cast(valid_accuracy, tf.float32))

        test_accuracy = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
        test_accuracy = tf.reduce_mean(tf.cast(test_accuracy, tf.float32))

        confusion_mat = tf.confusion_matrix(tf.argmax(model,1), tf.argmax(y,1))

        #training
        init = tf.global_variables_initializer()
        # monitoring
        l_sum = tf.summary.scalar("loss", loss)
        t_sum = tf.summary.scalar("training_accuracy", train_accuracy)
        v_sum = tf.summary.scalar("validation_accuracy", valid_accuracy)
        merged_summary = tf.summary.merge([l_sum, t_sum])

        #session
        with tf.Session() as sess:
            sess.run(init)
            #writer
            summary_writer = tf.summary.FileWriter(config.LOG_PATH, graph=tf.get_default_graph())

            #train
            nb_batch = int(data.train.num_examples/batch_size)
            for epoch in range(training_epochs):
                for batch in range(nb_batch):
                    batch_xs, batch_ys = data.train.next_batch(batch_size)
                    step = epoch*nb_batch+batch
                    if step % measure_frequency == 0:
                        t_acc = train_accuracy.eval(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                        v_acc, v_summary = sess.run([valid_accuracy, v_sum], feed_dict={x: data.validation.signals, y: data.validation.labels, keep_prob: 1.0})
                        summary_writer.add_summary(v_summary, step)
                        print('step {0}, training accuracy {1:.4f}, validation accuracy {2:.4f}'.format(step, t_acc, v_acc))
                    _, l, summary = sess.run([optimizer, loss, merged_summary], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
                    summary_writer.add_summary(summary, step)
            #model accuracyt
            test_acc, confusion = sess.run([test_accuracy, confusion_mat], feed_dict={x: data.test.signals, y: data.test.labels, keep_prob: 1.0})
            print(test_acc)
            heatmap(summary_writer, confusion, data.labels)
            summary_writer.close()

def heatmap(writer, confusion_matrix, labels):
    plt.figure()
    confusion_matrix[(confusion_matrix > 0)] = 1

    fig, ax = plt.subplots()
    ax.pcolor(confusion_matrix, cmap=plt.cm.Blues, alpha=0.8)
    ax.set_frame_on(False)
    ax.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(labels, fontsize=2)
    ax.set_yticklabels(labels, fontsize=2)
    plt.xticks(rotation=90)
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=400)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    summary = tf.summary.image("test", image, max_outputs=1)
    writer.add_summary(summary.eval())
