import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import FeatureSpace
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import seaborn as sns

keras.utils.set_random_seed(42)
print("Reading data...")

data = pd.read_csv('dropout.csv', delimiter=';')
data.loc[data["Target"] == "Graduate", "Target"] = "Enrolled"


# --------------------------
# PREPROCESSING<br>
# --------------------------


data_val = data.sample(frac=0.2, random_state=42)
data_train = data.drop(data_val.index)

def dataframe_to_dataset(df):
    df = df.copy()
    labels_str = df.pop('Target')
    labels = pd.get_dummies(labels_str)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    #ds = ds.shuffle(buffer_size=len(df))
    return ds

def dataframe_to_dataset_balanced(df):
    df = df.copy()
    labels_str = df.pop('Target')
    labels = pd.get_dummies(labels_str)
    ds1 = tf.data.Dataset.from_tensor_slices((dict(df[labels_str == "Enrolled"]), labels[labels_str=="Enrolled"]))
    ds2 = tf.data.Dataset.from_tensor_slices((dict(df[labels_str == "Dropout"]), labels[labels_str=="Dropout"]))
    ds3 = tf.data.Dataset.from_tensor_slices((dict(df[labels_str == "Graduate"]), labels[labels_str=="Graduate"]))

    resampled = tf.data.Dataset.sample_from_datasets([ds1, ds2, ds3], weights=[0.33, 0.33, 0.33])
    return resampled


ds_val = dataframe_to_dataset(data_val)
ds_train = dataframe_to_dataset(data_train)
#ds_train = dataframe_to_dataset_balanced(data_train)


ds_val = ds_val.batch(64)
ds_train = ds_train.batch(64)

print("Preprocessing data...")

feature_space = FeatureSpace(
    features={
        "Marital status": FeatureSpace.integer_categorical(),
        "Application mode": FeatureSpace.integer_categorical(),
        "Application order": FeatureSpace.float_rescaled(),
        "Course": FeatureSpace.integer_categorical(),
        "Daytime/evening attendance\t": FeatureSpace.integer_categorical(),
        "Previous qualification" : FeatureSpace.integer_categorical(),
        "Previous qualification (grade)" : FeatureSpace.float_normalized(),                                  # try float_rescaled
        "Nacionality" : FeatureSpace.integer_categorical(),
        "Mother's qualification" : FeatureSpace.integer_categorical(),
        "Father's qualification" : FeatureSpace.integer_categorical(),
        "Mother's occupation" : FeatureSpace.integer_categorical(),
        "Father's occupation" : FeatureSpace.integer_categorical(),
        "Admission grade" : FeatureSpace.float_normalized(),                                                 # try float_rescaled
        "Displaced" : FeatureSpace.integer_categorical(),
        "Educational special needs" : FeatureSpace.integer_categorical(),
        "Debtor" : FeatureSpace.integer_categorical(),
        "Tuition fees up to date" : FeatureSpace.integer_categorical(),
        "Gender" : FeatureSpace.integer_categorical(),
        "Scholarship holder" : FeatureSpace.integer_categorical(),
        "Age at enrollment" : FeatureSpace.float_normalized(),                                               # maybe float_discretized
        "International": FeatureSpace.integer_categorical(),
        "Curricular units 1st sem (credited)" : FeatureSpace.float_normalized(),
        "Curricular units 1st sem (enrolled)" : FeatureSpace.float_normalized(),
        "Curricular units 1st sem (evaluations)" : FeatureSpace.float_normalized(),
        "Curricular units 1st sem (approved)" : FeatureSpace.float_normalized(),
        "Curricular units 1st sem (grade)" : FeatureSpace.float_normalized(),
        "Curricular units 1st sem (without evaluations)" : FeatureSpace.float_normalized(),
        "Curricular units 2nd sem (credited)" : FeatureSpace.float_normalized(),
        "Curricular units 2nd sem (enrolled)" : FeatureSpace.float_normalized(),
        "Curricular units 2nd sem (evaluations)" : FeatureSpace.float_normalized(),
        "Curricular units 2nd sem (approved)" : FeatureSpace.float_normalized(),
        "Curricular units 2nd sem (grade)" : FeatureSpace.float_normalized(),
        "Curricular units 2nd sem (without evaluations)" : FeatureSpace.float_normalized(),
        "Unemployment rate" : FeatureSpace.float_normalized(),
        "Inflation rate" : FeatureSpace.float_normalized(),
        "GDP" : FeatureSpace.float_normalized(),
    },
    output_mode="concat"
)


ds_train_no_labels = ds_train.map(lambda x, _:x)
feature_space.adapt(ds_train_no_labels)


ds_train_preprocessed = ds_train.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
ds_train_preprocessed = ds_train_preprocessed.prefetch(tf.data.AUTOTUNE)

ds_val_preprocessed = ds_val.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
ds_val_preprocessed = ds_val_preprocessed.prefetch(tf.data.AUTOTUNE)

dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

def get_model(hp):
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.L2(1e-4), bias_regularizer=keras.regularizers.L2(1e-4), kernel_initializer = keras.initializers.GlorotNormal())(encoded_features)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.L2(1e-4), bias_regularizer=keras.regularizers.L2(1e-4), kernel_initializer = keras.initializers.GlorotNormal())(x)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(2, activation="softmax")(x)


# class weights are only valuable if we want to move precision/recall
# it basically assigns more samples to the underrepresented class
    from sklearn.utils.class_weight import compute_class_weight

    y_integers = np.argmax(np.concatenate([y for _,y in ds_train], axis=0), axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
    d_class_weights = dict(enumerate(class_weights))
    d_class_weights

    model = keras.Model(inputs=encoded_features, outputs=predictions)
    model.compile(
        optimizer="adam", loss="binary_crossentropy", 
            metrics=[
                "accuracy",
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                #keras.metrics.F1Score(name='f1'),
            ]
    )

    return model

model = get_model()


# In[426]:


history = model.fit(
    ds_train_preprocessed, 
    epochs=15, 
    validation_data=ds_val_preprocessed, 
    verbose=1,
    #class_weight=d_class_weights
)

def plot_metrics(history):
    plt.plot(history.epoch, history.history['val_accuracy'], label="Accuracy")
    plt.plot(history.epoch, history.history['val_precision'], label="Precision")
    plt.plot(history.epoch, history.history['val_recall'], label="Recall")
    
    plt.plot(history.epoch, history.history['accuracy'], label="Accuracy (train)")
    plt.plot(history.epoch, history.history['precision'], label="Precision (train)")
    plt.plot(history.epoch, history.history['recall'], label="Recall (train)")
    #plt.plot(history.epoch, history.history['val_f1'], label="F1")
    plt.ylim([0.5,1])
    plt.legend()
    plt.show()

def plot_confusion(ds_val):
    predictions = np.argmax(model.predict(ds_val_preprocessed), axis=1)
    trues = np.argmax(np.concatenate([y for _,y in ds_val], axis=0), axis=1)
    confusion = confusion_matrix(predictions, trues) 
    s = sns.heatmap(confusion, annot=True, fmt="3d", cmap="viridis")
    s.set(xlabel="predicted", ylabel="true")
    plt.show()
    print("F1:", f1_score(trues, predictions, average='macro'))

plot_metrics(history)
plot_confusion(ds_val)
