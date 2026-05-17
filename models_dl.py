"""
Deep learning model definitions: CNN (corresponds to paper Section 4.3 Figure 4.3) and BP neural network.
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers

from config import CNN_INPUT_SHAPE, NUM_CLASSES, EPOCHS, BATCH_SIZE


def build_cnn_model(input_shape=CNN_INPUT_SHAPE, num_classes=NUM_CLASSES,
                    conv_filters=(8, 16), dense_units=(256, 128)):
    """Build a CNN model.

    The default architecture follows the paper's Figure 4.3:
        Conv2D(8, 3x3) -> MaxPool(2x2) -> Conv2D(16, 3x3) -> MaxPool(2x2)
        -> Flatten -> Dense(256) -> Dense(128) -> Dense(12, softmax)
    """
    model = Sequential([
        layers.Conv2D(filters=conv_filters[0], kernel_size=(3, 3),
                      padding='same', input_shape=input_shape, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=conv_filters[1], kernel_size=(3, 3),
                      padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        layers.Flatten(),
        layers.Dense(dense_units[0], activation='relu'),
        layers.Dense(dense_units[1], activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


def build_cnn1d_model(input_dim=256, num_classes=NUM_CLASSES,
                      conv_filters=(8, 16), kernel_size=5, pool_size=4,
                      dense_units=(256, 128)):
    """Build a 1D CNN model.

    Motivation: Moore features are statistics in a linear order (port, packet count,
    packet size, time, flags). Reshaping to 16x16 introduces artificial 2D
    neighborhoods that are not semantically meaningful. 1D conv preserves the
    original sequential locality.

    Default: Conv1D(8, k=5) -> MaxPool1D(p=4) -> Conv1D(16, k=5) -> MaxPool1D(p=4)
             -> Flatten -> Dense(256) -> Dense(128) -> Dense(12, softmax)
    The number of channels matches the 2D version for a fair parameter-count comparison;
    kernel 5 covers 5 adjacent features (similar receptive field to 2D 3x3 along one axis).
    """
    model = Sequential([
        layers.Conv1D(filters=conv_filters[0], kernel_size=kernel_size,
                      padding='same', input_shape=(input_dim, 1), activation='relu'),
        layers.MaxPooling1D(pool_size=pool_size, padding='same'),
        layers.Conv1D(filters=conv_filters[1], kernel_size=kernel_size,
                      padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=pool_size, padding='same'),
        layers.Flatten(),
        layers.Dense(dense_units[0], activation='relu'),
        layers.Dense(dense_units[1], activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


def build_cnn1d_dilated_model(input_dim=256, num_classes=NUM_CLASSES,
                              filters=(16, 32, 32), kernel_size=3,
                              dilations=(1, 2, 4), dense_units=(256, 128)):
    """Build a 1D CNN with dilated convolutions.

    Motivation: an ordinary Conv1D with k=5 has a receptive field of only 5
    adjacent features per layer. With dilation, each successive layer's
    receptive field grows exponentially without adding parameters:
        layer1 (d=1): receptive field 3
        layer2 (d=2): receptive field 3+(3-1)*2 = 7
        layer3 (d=4): receptive field 7+(3-1)*4 = 15
    A 15-element receptive field can simultaneously capture port + packet count +
    size + flags - several semantic groups at once.

    Padding stays 'same' to keep the spatial dimension intact, so dilation only
    boosts receptive field, never sequence length.
    """
    inp = layers.Input(shape=(input_dim, 1))
    x = inp
    for f, d in zip(filters, dilations):
        x = layers.Conv1D(filters=f, kernel_size=kernel_size,
                          padding='same', dilation_rate=d,
                          activation='relu')(x)
    # Reduce spatial dim while keeping all channels
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units[0], activation='relu')(x)
    x = layers.Dense(dense_units[1], activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


def build_bp_model(input_dim=256, num_classes=NUM_CLASSES,
                   hidden_units=(256, 128, 64)):
    """Build a fully-connected BP neural network.

    Architecture: Input -> Dense(256) -> Dense(128) -> Dense(64) -> Dense(12, softmax)
    Compared to CNN, this lacks the convolutional + pooling structure.
    """
    model = Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(hidden_units[0], activation='sigmoid'),
        layers.Dense(hidden_units[1], activation='sigmoid'),
        layers.Dense(hidden_units[2], activation='sigmoid'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


def train_cnn(X_train, y_train, X_test, y_test,
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1):
    """Train the CNN model and return (model, history, test accuracy, time spent, predictions)."""
    X_train_r = tf.reshape(tf.cast(X_train, tf.float32), [-1, 16, 16, 1])
    X_test_r = tf.reshape(tf.cast(X_test, tf.float32), [-1, 16, 16, 1])

    model = build_cnn_model()
    model.summary()

    t1 = time.time()
    history = model.fit(
        X_train_r, y_train,
        validation_data=(X_test_r, y_test),
        epochs=epochs, batch_size=batch_size,
        verbose=verbose,
    )
    t2 = time.time()

    scores = model.evaluate(X_test_r, y_test, verbose=0)
    pred = model.predict(X_test_r, verbose=0)
    pred_labels = np.argmax(pred, axis=1)

    elapsed = t2 - t1
    print(f'[CNN] Test accuracy: {scores[1]:.4f}, time spent: {elapsed:.2f}s')

    return {
        'model': model,
        'history': history.history,
        'accuracy': scores[1],
        'loss': scores[0],
        'time': elapsed,
        'pred': pred_labels,
    }


def train_cnn1d(X_train, y_train, X_test, y_test,
                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                kernel_size=5, pool_size=4):
    """Train the 1D CNN model."""
    X_train_r = tf.reshape(tf.cast(X_train, tf.float32), [-1, X_train.shape[1], 1])
    X_test_r = tf.reshape(tf.cast(X_test, tf.float32), [-1, X_test.shape[1], 1])

    model = build_cnn1d_model(input_dim=X_train.shape[1],
                              kernel_size=kernel_size, pool_size=pool_size)
    model.summary()

    t1 = time.time()
    history = model.fit(
        X_train_r, y_train,
        validation_data=(X_test_r, y_test),
        epochs=epochs, batch_size=batch_size,
        verbose=verbose,
    )
    t2 = time.time()

    scores = model.evaluate(X_test_r, y_test, verbose=0)
    pred = model.predict(X_test_r, verbose=0)
    pred_labels = np.argmax(pred, axis=1)

    elapsed = t2 - t1
    print(f'[CNN1D] Test accuracy: {scores[1]:.4f}, time spent: {elapsed:.2f}s')

    return {
        'model': model,
        'history': history.history,
        'accuracy': scores[1],
        'loss': scores[0],
        'time': elapsed,
        'pred': pred_labels,
    }


def train_cnn1d_dilated(X_train, y_train, X_test, y_test,
                       epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1):
    """Train the dilated 1D CNN."""
    X_train_r = tf.reshape(tf.cast(X_train, tf.float32), [-1, X_train.shape[1], 1])
    X_test_r = tf.reshape(tf.cast(X_test, tf.float32), [-1, X_test.shape[1], 1])

    model = build_cnn1d_dilated_model(input_dim=X_train.shape[1])
    model.summary()

    t1 = time.time()
    history = model.fit(
        X_train_r, y_train,
        validation_data=(X_test_r, y_test),
        epochs=epochs, batch_size=batch_size,
        verbose=verbose,
    )
    t2 = time.time()

    scores = model.evaluate(X_test_r, y_test, verbose=0)
    pred = model.predict(X_test_r, verbose=0)
    pred_labels = np.argmax(pred, axis=1)

    elapsed = t2 - t1
    print(f'[CNN1D-Dilated] Test accuracy: {scores[1]:.4f}, time spent: {elapsed:.2f}s')

    return {
        'model': model,
        'history': history.history,
        'accuracy': scores[1],
        'loss': scores[0],
        'time': elapsed,
        'pred': pred_labels,
    }


def train_bp(X_train, y_train, X_test, y_test,
             epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1):
    """Train the BP neural network."""
    X_train_f = tf.cast(X_train, tf.float32)
    X_test_f = tf.cast(X_test, tf.float32)

    model = build_bp_model(input_dim=X_train.shape[1])
    model.summary()

    t1 = time.time()
    history = model.fit(
        X_train_f, y_train,
        validation_data=(X_test_f, y_test),
        epochs=epochs, batch_size=batch_size,
        verbose=verbose,
    )
    t2 = time.time()

    scores = model.evaluate(X_test_f, y_test, verbose=0)
    pred = model.predict(X_test_f, verbose=0)
    pred_labels = np.argmax(pred, axis=1)

    elapsed = t2 - t1
    print(f'[BP] Test accuracy: {scores[1]:.4f}, time spent: {elapsed:.2f}s')

    return {
        'model': model,
        'history': history.history,
        'accuracy': scores[1],
        'loss': scores[0],
        'time': elapsed,
        'pred': pred_labels,
    }
