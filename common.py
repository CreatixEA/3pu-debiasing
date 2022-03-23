import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import datetime
import os
from tensorflow import keras

import logging
import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)

def load_data(name='ancien_prepared'):
    '''
    Load dataset and return the  df
    '''
    try:
        df = pd.read_csv(f'data/{name}.csv', index_col=0)
        type_columns(df)
        log.info(f"Dataset {name} loaded and typed")
    except FileNotFoundError as e:
        raise FileNotFoundError("You should pay attention to the 'name' argument of the load_data function")

    return df

def type_columns(df):
    '''
    Type all columns depending of the configuration
    The configuration can be found in the 'archived/columns_description.py'
    '''
    
    from archived.columns_description import cols_categorical, cols_numerical
    for col in cols_categorical:
        df[col] = df[col].astype('category')
    
    for col in cols_numerical:
        df[col] = pd.to_numeric(df[col], downcast='float')

def split_dataset(df):
    '''
    Split a dataset from the validation_indexes file
    '''
    with open('archived/validation_indexes.pkl', 'rb') as f:
        validation_indexes = pickle.load(f)

    validation_df = df.loc[validation_indexes].copy()
    train_df = df.drop(validation_indexes).copy()

    return train_df, validation_df
def load_data_splited(name='ancien_prepared'):
    '''
    Load dataset and split using the split indexes saved in archives
    '''
    df = load_data(name)
    return split_dataset(df)

def load_missing_data():
    '''
    Load missing dataset
    '''
    return pd.read_csv('archived/missing_values_matrix.csv', index_col=0)


def generate_model(train_df, validation_df):
    '''
    Generate a standard model using the df
    '''
    from archived.columns_description import target

    X_train = train_df.drop(target, axis=1).copy()
    Y_train = train_df[target].copy()

    X_val = validation_df.drop(target, axis=1).copy()
    Y_val = validation_df[target].copy()

    dtrain = xgb.DMatrix(X_train, Y_train, enable_categorical=True)
    bst = xgb.train({'max_depth': 6, 'objective': 'binary:logistic', 'tree_method': 'hist'},  dtrain, num_boost_round=10)

    predictions = bst.predict(xgb.DMatrix(X_val, enable_categorical=True))
    Y_final = Y_val.apply(lambda x: 1 if x else 0).astype(float).to_numpy()    
    
    auc = roc_auc_score(Y_final, predictions)
    log.info(f"AUC: {auc}")
    fpr, tpr, threshold = roc_curve(Y_final, predictions)

    return {
        'auc': auc,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'threshold': threshold.tolist(),
        'model': bst
    }

def generate_deep_model(train_df, validation_df):
    '''
    Generate a deep model
    '''
    from archived.columns_description import cols_categorical, target

    targets = target

    if not os.path.exists('data/tmp.pkl') or True: # Always do that
        for column in cols_categorical:
            if len(train_df[column].unique()) <= 2: # It can be represented by a 1:0 couple
                first = train_df[column].unique()[0]
                train_df[column] = train_df[column].apply(lambda x: 1 if x else 0)

                validation_df[column] = validation_df[column].apply(lambda x: 1 if x else 0)

                log.info(f"{column} binary encoded")
                continue

            lookup = keras.layers.StringLookup(output_mode='one_hot')

            lookup.adapt(train_df[column].astype('str'))

            encoded = lookup(train_df[column].astype('str'))
            columns = [f'{column}_{i}' for i in range(len(encoded[0]))]
            train_df = pd.merge(train_df.drop(column, axis=1), pd.DataFrame(encoded, index=train_df.index, columns=columns), left_index=True, right_index=True)

            encoded = lookup(validation_df[column].astype('str'))
            validation_df = pd.merge(validation_df.drop(column, axis=1), pd.DataFrame(encoded, index=validation_df.index, columns=columns), left_index=True, right_index=True)

            if column == target: # Then we store encoded columns in the targets
                targets = columns

            log.info(f"{column} one-hot encoded, resulting {train_df.shape} and {validation_df.shape}")
        tmp = (train_df, validation_df, targets)
        pickle.dump(tmp, open('data/tmp.pkl', 'wb'))
    else:
        train_df, validation_df, targets = pickle.load(open('data/tmp.pkl', 'rb'))

    X_train = train_df.drop(targets, axis=1).copy()
    Y_train = pd.DataFrame(train_df[targets].copy())

    X_val = validation_df.drop(targets, axis=1).copy()
    Y_val = pd.DataFrame(validation_df[targets].copy())

    inputs = keras.Input(shape=(X_train.shape[1], ))

    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    x = keras.layers.Dense(4, activation="relu")(x)

    output_dimensions = Y_train.shape[1] if len(Y_train.shape) > 1 else 1
    outputs = keras.layers.Dense(output_dimensions)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='deep_model')
    model.summary()

    try:
        keras.utils.plot_model(model, to_file='archived/deep_model.png', show_shapes=True, show_layer_activations=True)
    except:
        log.warning("Cannot graph model")

    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    history = model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_split=0.2)
    log.info(f"History: \n{history}")

    test_scores = model.evaluate(X_val, Y_val, verbose=2)
    log.info(f"Test scores:\n{test_scores}")

    predictions = pd.DataFrame(model.predict(X_val))

    auc = roc_auc_score(Y_val, predictions)
    log.info(f"AUC: {auc}")
    fpr, tpr, threshold = roc_curve(Y_val, predictions)

    return {
        'auc': auc,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'threshold': threshold.tolist(),
        'model': None,
    }

def draw_roc(name, fpr, tpr, auc, **kwargs):
    '''
    Draw the ROC
    '''
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4}")
    plt.title(f"ROC for {name}\nE. ARNAUD - 3P-U debiaisor - {datetime.date.today()}")
    plt.xlabel("$1-Specificity$")
    plt.ylabel("$Sensitivity$")
    plt.legend()

    if 'save' in kwargs:
        plt.savefig(f"archived/model_{kwargs.get('save')}_roc.png")

    plt.show()

if __name__ == '__main__':
    print(load_data())
