import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math


def get_model():
    # network model
    model = keras.Sequential()
    model.add(layers.Conv2D(1, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(),
    #                         use_bias=True, bias_initializer=tf.keras.initializers.Ones(),
    #                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    # model.add(layers.Dense(128, use_bias=True, bias_initializer=tf.keras.initializers.Ones(),
    #                        kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='tanh'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model


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


def create_dataset(d, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((d, labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset


class SNR:
    def set(self, value):
        self.value = value
        self.accuracy = 0
        self.loss = 0


def add_snr(list_snr, value, accuracy, loss):
    for i in list_snr:
        if i.value == value:
            i.accuracy=(accuracy)
            i.loss=(loss)
    return list_snr


def get_snr_acc(list_snr, value):
    for i in list_snr:
        if i.value == value:
            return i.accuracy

def get_snr_loss(list_snr, value):
    for i in list_snr:
        if i.value == value:
            return i.loss


def awgn(SNRdB, model, inject_to_weights):
    # convert to snr
    temp = SNRdB / 10
    snr = 10 ** temp
    var_noises = []
    for layer in model.trainable_weights:
        n = np.random.standard_normal(layer.shape)
        es = np.sum(layer ** 2)
        en = np.sum(n ** 2)
        alpha = np.sqrt(es / (snr * en))
        if inject_to_weights:
            layer.assign_add(alpha * n)
        var_noises.append(np.var(alpha * n))
    return var_noises


class MySGDOptimizer(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, name="MySGDOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)  # handle learning rate decay
        #         momentum_var = self.get_slot(var, "momentum")
        #         momentum_hyper = self._get_hyper("momentum", var_dtype)
        #         momentum_var.assign(momentum_var * momentum_hyper - (1. - momentum_hyper)* grad)

        var.assign(var)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config
        }


class Taylorcallback(keras.callbacks.Callback):
    def __init__(self, model, snr):
        super(Taylorcallback, self).__init__()
        self.model = model
        self.snr = snr

    def on_epoch_end(self, epoch, logs=None):
        print("Training accuracy taylor over epoch: %d : %.4f" % (epoch, float(logs["categorical_accuracy"]),))

    def on_train_batch_begin(self, batch, logs=None):
        print("on_train_batch_begin")
        mu = 0.001
        new_weights = []
        var_noises = awgn(self.snr, self.model, False)
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape() as tape1:
                logits = self.model(data_list[batch][0], training=True)
                loss_value = loss_fn(data_list[batch][1], logits)
                if math.isnan(loss_value):
                    print("loss is nan")
            grads = tape1.gradient(loss_value, self.model.trainable_weights)
        for idx, val in enumerate(grads):
            h = tape2.jacobian(val, self.model.trainable_weights[idx], experimental_use_pfor=False)
            var_noise = var_noises[idx]
            # updating weights based on taylor expansion
            # w^k=w^(k-1)-μ[∇_w l(B;w)+1/2 σ^2 ∇_w^2 l(B;w) ∇_w l(B;w)]
            # reshaping for matrix multiplication
            h_linear = np.reshape(h, [-1])
            d_h_linear = np.sqrt(h_linear.shape)
            h_reshape = np.reshape(h, (np.int(d_h_linear[0]), np.int(d_h_linear[0])))
            val_linear = np.reshape(val, [-1])
            d_val_linear = val_linear.shape
            val_reshape = np.reshape(val, (np.int(np.array(d_val_linear)[0])))
            matmul = np.matmul(h_reshape, val_reshape)
            temp = np.array(val, dtype=np.float64) + 1 / 2 * var_noise * np.reshape(matmul, val.shape)
            new_weights.append(self.model.trainable_weights[idx] - mu * temp)
            if np.isnan(val).any():
                print("grads is nan")
            if np.isnan(h).any():
                print("hessian is nan")
        self.model.set_weights(new_weights)

    def on_test_batch_begin(self, batch, logs=None):
        awgn(self.snr, self.model, True);
class Noiseawarecallback(keras.callbacks.Callback):
    def __init__(self, model, snr):
        super(Noiseawarecallback, self).__init__()
        self.model = model
        self.snr = snr

    def on_epoch_end(self, epoch, logs=None):
        print("Training accuracy SGD noise aware over epoch: %d : %.4f" % (epoch, float(logs["categorical_accuracy"]),))

    def on_train_batch_begin(self, batch, logs=None):
        awgn(self.snr, self.model, True)

    def on_test_batch_begin(self, batch, logs=None):
        awgn(self.snr, self.model, True);
class Noiseunawarecallback(keras.callbacks.Callback):
    def __init__(self, model, snr):
        super(Noiseunawarecallback, self).__init__()
        self.model = model
        self.snr = snr

    def on_epoch_end(self, epoch, logs=None):
        print("Training accuracy SGD over epoch: %d : %.4f" % (epoch, float(logs["categorical_accuracy"]),))
    def on_test_batch_begin(self, batch, logs=None):
        awgn(self.snr, self.model, True);

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255, test_images / 255

y_train_one_hot = to_categorical(train_labels)
y_test_one_hot = to_categorical(test_labels)
#
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
batch_size = 64
data = create_dataset(train_images[0:105], y_train_one_hot[0:105], batch_size)
dataset_test = create_dataset(test_images[0:105], y_test_one_hot[0:105], batch_size)
data_list = list(data)
SNRs = [25]
output_taylor = {}
output_SGD_noise_aware = {}
output_SGD_noise_unaware = {}
output_taylor['test'] = []
output_SGD_noise_aware['test'] = []
output_SGD_noise_unaware['test'] = []
#
for snr in SNRs:
    print("taylor is using for SNR " + str(snr))
    a = SNR()
    a.set(snr)
    output_taylor['test'].append(a)
    model_taylor = get_model()
    model_taylor.compile(loss="categorical_crossentropy", optimizer=MySGDOptimizer(), metrics=["categorical_accuracy"])
    model_taylor.fit(train_images[0:105], y_train_one_hot[0:105], epochs=1, batch_size=batch_size,validation_split=0.3,
                     callbacks=[Taylorcallback(model_taylor, snr), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0,mode='min')])
    #end of taylor training
    #test
    loss, acc = model_taylor.evaluate(test_images[0:105], y_test_one_hot[0:105],verbose=0, batch_size=batch_size,callbacks=[Taylorcallback(model_taylor, snr)])
    add_snr(output_taylor['test'], snr, acc, loss);
###########################################################
for snr in SNRs:
    print("SGD noise aware is using for SNR " + str(snr))
    a = SNR()
    a.set(snr)
    output_SGD_noise_aware['test'].append(a)
    model_SGD_noise_aware = get_model()
    model_SGD_noise_aware.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["categorical_accuracy"])
    model_SGD_noise_aware.fit(train_images[0:105], y_train_one_hot[0:105], epochs=1, batch_size=batch_size,validation_split=0.3,
                     callbacks=[Noiseawarecallback(model_SGD_noise_aware, snr), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0,mode='min')])
    #end of taylor training
    #test
    loss, acc = model_SGD_noise_aware.evaluate(test_images[0:105], y_test_one_hot[0:105],verbose=0, batch_size=batch_size,callbacks=[Noiseawarecallback(model_SGD_noise_aware, snr)])
    add_snr(output_SGD_noise_aware['test'], snr, acc, loss);
##########################################################
for snr in SNRs:
    print("SGD noise unaware is using for SNR " + str(snr))
    a = SNR()
    a.set(snr)
    output_SGD_noise_unaware['test'].append(a)
    model_SGD_noise_unaware = get_model()
    model_SGD_noise_unaware.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["categorical_accuracy"])
    model_SGD_noise_unaware.fit(train_images[0:105], y_train_one_hot[0:105], epochs=1, batch_size=batch_size,validation_split=0.3,
                     callbacks=[Noiseunawarecallback(model_SGD_noise_unaware, snr)])
    #end of taylor training
    #test
    loss, acc = model_SGD_noise_unaware.evaluate(test_images[0:105], y_test_one_hot[0:105],verbose=0, batch_size=batch_size,callbacks=[Noiseunawarecallback(model_SGD_noise_unaware, snr), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0,mode='min')])
    add_snr(output_SGD_noise_unaware['test'], snr, acc, loss);
################
#figures
loss_taylor = [];
acc_taylor = [];
loss_noise_unaware = [];
acc_noise_unaware = [];
loss_noise_aware = [];
acc_noise_aware = [];
for snr in SNRs:
    loss_taylor.append(get_snr_loss(output_taylor["test"], snr));
    acc_taylor.append(get_snr_acc(output_taylor["test"], snr));
for snr in SNRs:
    loss_noise_unaware.append(get_snr_loss(output_SGD_noise_unaware["test"], snr));
    acc_noise_unaware.append(get_snr_acc(output_SGD_noise_unaware["test"], snr));
for snr in SNRs:
    loss_noise_aware.append(get_snr_loss(output_SGD_noise_aware["test"], snr));
    acc_noise_aware.append(get_snr_acc(output_SGD_noise_aware["test"], snr));
create_plot_y(SNRs, acc_taylor, "Accuracy while using taylor expansion")
create_plot_y(SNRs, acc_noise_aware, "Accuracy while using SGD and noise")
create_plot_y(SNRs, acc_noise_unaware, "Accuracy while using SGD and no noise")
display_plot("Comparison of accuracy", "SNR", "Accuracy")
create_plot_y(SNRs, loss_taylor, "Loss while using taylor expansion")
create_plot_y(SNRs, loss_noise_aware, "Loss while using SGD and noise")
create_plot_y(SNRs, loss_noise_unaware, "Loss while using SGD and no noise")
display_plot("Comparison of loss", "SNR", "Loss")
#pip install pyqt5
