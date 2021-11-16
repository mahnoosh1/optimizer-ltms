import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import gc
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

##########
tf.config.experimental.set_lms_enabled(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


#########
def get_model():
    # network model
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', use_bias=True, input_shape=(32, 32, 3)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', use_bias=True))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, use_bias=True, activation='tanh'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, use_bias=True, activation='softmax'))
    return model


def awgn(SNRdB, model, inject_to_weights):
    # convert to snr
    temp = SNRdB / 10
    snr = 10 ** temp
    var_noises = []
    for layer in model.trainable_weights:
        n = np.random.standard_normal(layer.shape)
        es = tf.math.reduce_sum(layer ** 2)
        en = tf.math.reduce_sum(n ** 2)
        dim = tf.cast((snr * en), tf.float32)
        alpha = tf.sqrt(es / dim)
        if inject_to_weights:
            layer.assign_add(alpha * n)
        var = tf.math.reduce_variance(alpha * n)
        var_noises.append(var)
    return var_noises


def create_plot(d, label):
    plt.plot(d, label=label)


def display_plot(title, x_label, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def create_plot_y(data_x, data_y, label):
    plt.plot(data_x, data_y, label=label)


class SNR:
    def set(self, value):
        self.value = value
        self.accuracy = 0
        self.loss = 0


def add_snr(list_snr, value, accuracy, loss):
    for i in list_snr:
        if i.value == value:
            i.accuracy = (accuracy)
            i.loss = (loss)
    return list_snr


def train_step(inputs):
    images, labels = inputs
    new_weights = []
    var_noises = awgn(snr_current, model_taylor, False)
    mu = 0.01
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape() as tape1:
            predictions = model_taylor(images, training=True)
            loss = compute_loss(labels, predictions)
        grads = tape1.gradient(loss, model_taylor.trainable_weights)
    for idx, val in enumerate(grads):
        h = tape2.jacobian(val, model_taylor.trainable_weights[idx], experimental_use_pfor=False)
        # updating weights based on taylor expansion
        # w^k=w^(k-1)-μ[∇_w l(B;w)+1/2 σ^2 ∇_w^2 l(B;w) ∇_w l(B;w)]
        # reshaping for matrix multiplication
        h_linear = tf.reshape(h, [-1])
        d_h_linear = math.sqrt(h_linear.shape.as_list()[0])
        h_reshape = tf.reshape(h, [np.int(d_h_linear), np.int(d_h_linear)])
        val_linear = tf.reshape(val, [-1])
        d_val_linear = val_linear.shape.as_list()[0]
        val_reshape = tf.reshape(val, [d_val_linear, 1])
        matmul = tf.matmul(h_reshape, val_reshape)
        temp = val + 1 / 2 * tf.cast(var_noises[idx], tf.float32) * tf.reshape(matmul, val.shape)
        new_weights.append(tf.convert_to_tensor(model_taylor.trainable_weights[idx] - mu * temp))
    train_accuracy.update_state(labels, predictions)
    return new_weights,loss


def test_step(inputs):
    images, labels = inputs

    predictions = model_taylor(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)


@tf.function
def distributed_train_step(dataset_inputs):
    w,per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
    w_total=[]
    for per_replica_w in w:
        w_new=strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_w,
                        axis=None)
        w_total.append(w_new)
    return w_total, strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)


@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))


#############
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / np.float32(255), test_images / np.float32(255)

y_train_one_hot = to_categorical(train_labels)
y_test_one_hot = to_categorical(test_labels)
SNRs = [5, 10, 15, 20, 25]
strategy = tf.distribute.MirroredStrategy()
BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 50
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, y_train_one_hot)).shuffle(BUFFER_SIZE).batch(
    GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, y_test_one_hot)).batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

with strategy.scope():
    loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)


    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='train_accuracy')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='test_accuracy')

with strategy.scope():
    model_taylor = get_model()

print("number of replicas " + str(strategy.num_replicas_in_sync))
for snr in SNRs:
    snr_current = snr
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_w = []
        num_batches = 0
        for x in train_dist_dataset:
            new_weights = [];
            w_total,l = distributed_train_step(x)
            total_loss += l
            model_taylor.set_weights(w_total)
            print("set weight is done for batch "+str(num_batches))
            num_batches += 1
        train_loss = total_loss / num_batches
        train_accuracy.reset_states()

        # TEST LOOP
    for x in test_dist_dataset:
        distributed_test_step(x)
    template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                "Test Accuracy: {}")
    print(template.format(epoch + 1, train_loss,
                          train_accuracy.result() * 100, test_loss.result(),
                          test_accuracy.result() * 100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
