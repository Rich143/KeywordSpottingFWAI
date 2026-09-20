import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape, num_labels, initial_lr, alpha, conv_filters, dense_units, decay_steps):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=alpha,
    )

    max_pool_size = (2, 2)
    if dense_units > 32:
        max_pool_size = (4, 4)
        
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape(input_shape + (1,)),  # (30, 45) -> (30, 45, 1)

        # Convolutional layer (bias is set to False because BatchNorm adds its own beta bias parameter)
        layers.Conv2D(filters=conv_filters, kernel_size=(3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        # Non-linear activation (applied after batch normalization)
        layers.Activation('relu'),


            layers.MaxPooling2D(pool_size=max_pool_size),

        layers.Flatten(),

        layers.Dense(units=dense_units, activation='relu'),

        layers.Dense(num_labels),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


def objective(trial, train_ds, val_ds, label_names, epochs=100):
    initial_lr = trial.suggest_float('initial_lr', 1e-4, 3e-3, log=True)
    alpha = trial.suggest_float('alpha', 0.001, 0.1, log=True)
    conv_filters = trial.suggest_categorical('conv_filters', [16, 24, 32, 48, 64])
    dense_units = trial.suggest_categorical('dense_units', [8, 16, 32, 48, 64])

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    decay_steps = epochs * steps_per_epoch

    for example_spectrograms, example_spect_labels in train_ds.take(1):
        break
    input_shape = example_spectrograms.shape[1:]
    num_labels = len(label_names)

    model = build_model(
        input_shape=input_shape,
        num_labels=num_labels,
        initial_lr=initial_lr,
        alpha=alpha,
        conv_filters=conv_filters,
        dense_units=dense_units,
        decay_steps=decay_steps,
    )

    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[pruning_callback, early_stop],
        verbose=0,
    )

    val_accuracy = max(history.history['val_accuracy'])

    # Track model size alongside accuracy for later filtering (STM32U5 deployment budget)
    trial.set_user_attr('param_count', model.count_params())

    return val_accuracy


def run_study(train_ds, val_ds, label_names, n_trials=50, epochs=100):
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=30,
                interval_steps=5,
            ),
            patience=15,
        ),
    )
    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds, label_names, epochs=epochs),
        n_trials=n_trials,
    )

    print('Best trial:')
    print(f'  Value: {study.best_trial.value}')
    print(f'  Params: {study.best_trial.params}')
    print(f'  Param count: {study.best_trial.user_attrs.get("param_count")}')

    return study


study = run_study(train_ds, val_ds, label_names, n_trials=500, epochs=100)
