
# --------------------------
# IMPORTS
# --------------------------

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import FeatureSpace

print("Reading data...")

# --------------------------
# LOAD DATASET
# --------------------------
data = pd.read_csv('dropout.csv', delimiter=';')

# --------------------------
# PREPROCESSING
# --------------------------

# Split into validation and training dataset
data_val = data.sample(frac=0.2, random_state=42)
data_train = data.drop(data_val.index)

def dataframe_to_dataset(df):
    df = df.copy()
    labels_str = df.pop('Target')
    labels = pd.get_dummies(labels_str)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    df = ds.shuffle(buffer_size=len(df))
    return ds

ds_val = dataframe_to_dataset(data_val)
ds_train = dataframe_to_dataset(data_train)

ds_val = ds_val.batch(32)
ds_train = ds_train.batch(32)

# ideas for combinations (either by feature space or by hand)
# difference "curricular units credited - enrolled

# try out binning for age
# application order: treat it as a continous value

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

x = keras.layers.Dense(128, activation="relu")(encoded_features)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(32, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs=encoded_features, outputs=predictions)
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

history = model.fit(
    ds_train_preprocessed, epochs=10, validation_data=ds_val_preprocessed, verbose=2
)
