import fire

def setup():
    '''
    Start the setup processus. Must be run at least once
    '''
    import setup_env

def train_raw():
    '''
    Train the raw model with no modification and save results
    '''
    import common
    import pickle
    import logging

    logging.basicConfig()
    log = logging.getLogger()

    train, validation = common.load_data_splited()
    model = common.generate_model(train, validation)

    with open('archived/model_raw.pkl', 'wb') as f:
        pickle.dump(model, f)

    log.info("Model trained and saved")
    common.draw_roc("RAW", name='raw', **model)

def deep_model():
    import common
    import pickle
    from archived.columns_description import cols_categorical, cols_numerical
    log = common.logging.getLogger()

    df = common.load_data()

    for column in cols_categorical:
        df.loc[df[df[column].isna()].index, column] = df[column].mode()

    for column in cols_numerical:
        df.loc[df[df[column].isna()].index, column] = df[column].mean()

    train, validation = common.split_dataset(df)

    model = common.generate_deep_model(train, validation)
    
    with open('archived/model_deep_raw.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved")
    common.draw_roc("deep RAW", save='deep_raw', **model)

def train_mice(retrain=False):
    '''
    Train the MICE kernel and save it
    '''
    import mice
    import common
    import pickle
    import os
    import pandas as pd
    
    log = common.logging.getLogger()

    if not os.path.exists('archived/mice_kernel.pkl') or retrain:
        train, validation = common.load_data_splited()
        kernel = mice.train_mice(train)

        with open('archived/mice_kernel.pkl', 'wb') as f:
            pickle.dump(kernel, f)
        log.info("MICE KERNEL SAVED")
    else:
        kernel = common.load_mice_kernel()

    if not os.path.exists('data/ancien_miced.csv') or retrain:
        train_miced = mice.apply_mice(train, kernel)
        log.info("MICE applied to trained data")
        validation_miced = mice.apply_mice(validation, kernel)
        log.info("MICE applied to validation data")

        df_miced = pd.concat([train_miced, validation_miced])
        df_miced.to_csv('data/ancien_miced.csv')
        log.info("Ancien MICED dataset saved")
    else:
        train_miced, validation_miced = common.load_data_splited('ancien_miced')
        log.info("Ancien MICED loaded")


    model = common.generate_model(train_miced, validation_miced)

    with open('archived/model_mice.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved")
    common.draw_roc("MICE", save='mice', **model)

def correct_measures(retrain=False):
    import common
    import mice
    import os
    import pickle
    from archived.columns_description import target

    log = common.logging.getLogger()

    if not os.path.exists('data/ancien_miced_corrected.csv') or retrain:
        df = common.load_data('ancien_miced')
        missing_values = common.load_missing_data()
        kernel = mice.load_mice_kernel()

        # We delete values where data was initially not missing
        non_missing_values = (1 - missing_values) == True
        
        for column in df.columns:
            if column == target or all(non_missing_values[column]): # We exclude from the treatment the target value and the complete missing values
                log.info(f"I do not nullify on {column}")
                continue

            indexes = non_missing_values[non_missing_values[column] == True].index
            log.info(f"Nullify {column} {len(indexes)} non missing values")
            df.loc[indexes, column] = None

            # We enforce dtypes because of loading bug
            df[column] = df[column].astype(kernel.working_data[column].dtype)

        df = mice.apply_mice(df)
        df.to_csv('data/ancien_miced_corrected.csv')
        log.info("Corrected DF saved")
    else:
        df = common.load_data('ancien_miced_corrected')
    
    train, validation = common.split_dataset(df)
    model = common.generate_model(train, validation)

    with open('archived/model_mice_corrected.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved")
    common.draw_roc("MICE corrected", save='mice_corrected', **model)


def deep_model_mice():
    import common
    import pickle
    log = common.logging.getLogger()

    train, validation = common.load_data_splited('ancien_miced')

    model = common.generate_deep_model(train, validation)
    
    with open('archived/model_deep_mice.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved")
    common.draw_roc("deep MICE", save='deep_mice', **model)

def deep_model_corrected():
    import common
    import pickle
    log = common.logging.getLogger()

    train, validation = common.load_data_splited('ancien_miced_corrected')

    model = common.generate_deep_model(train, validation)
    
    with open('archived/model_deep_mice_corrected.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved")
    common.draw_roc("deep MICE corrected", save='deep_mice_corrected', **model)

def compare_datasets():
    import pandas as pd
    import common
    from archived.columns_description import cols_numerical
    log = common.logging.getLogger()

    miced = common.load_data('ancien_miced')
    corrected = common.load_data('ancien_miced_corrected')

    delta = pd.DataFrame()

    for column in cols_numerical:
        delta[column] = miced[column] - corrected[column]

    log.info("Description of Delta df")
    log.info(f"\n{delta.describe()}")

    delta.to_csv('data/delta.csv')

def compare_models():
    import pickle
    import os
    import common
    import re
    import matplotlib.pyplot as plt
    import datetime

    log = common.logging.getLogger()

    plt.figure(figsize=(10, 10))

    pattern = re.compile(r'model_(.+)\.pkl')

    models = {}

    for f in os.listdir('archived/'):
        m = pattern.match(f)
        if m:
            with open(f'archived/{f}', 'rb') as f:
                model = pickle.load(f)

            models[m.group(1)] = model
            

    for name, model in sorted(models.items(), key=lambda x: x[1]['auc']) :        
        plt.plot(model['fpr'], model['tpr'], label=f"{name} ({model['auc']:.4})")
            
    plt.title(f"ROC of all models\nE. ARNAUD - 3P-U debiaisor - {datetime.date.today()}")
    plt.legend(loc='lower right')
    plt.xlabel("$1-Specificity$")
    plt.ylabel("$Sensitivity$")

    plt.savefig('archived/all_models.png')

    plt.show()

if __name__ == '__main__':
    fire.Fire()
