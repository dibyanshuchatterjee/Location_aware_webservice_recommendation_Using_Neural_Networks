import argparse
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate
import load_data as F

# create argument parser object
parser = argparse.ArgumentParser()

# add required argument
parser.add_argument('df_depth', help='Number of rows of data to train with')

# parse arguments
args = parser.parse_args()

# Load the dataset
df = F.driver()
test_df = df[df['qos_response_time'] == 0]
train_df = df[df['qos_response_time'] != 0]
df = train_df[:int(args.df_depth)]

# Get unique values for each field
user_countries = df['user_country'].unique()
service_countries = df['service_country'].unique()
user_as = df['user_as'].unique()
service_as = df['service_as'].unique()
print("Got unique values")

# Create a dictionary for each field and its corresponding index
user_country_dict = {user_countries[i]: i for i in range(len(user_countries))}
service_country_dict = {service_countries[i]: i for i in range(len(service_countries))}
user_as_dict = {user_as[i]: i for i in range(len(user_as))}
service_as_dict = {service_as[i]: i for i in range(len(service_as))}
print("Dict created")

# Perform one-hot encoding on each field
user_country_one_hot = tf.one_hot(df['user_country'].apply(lambda x: user_country_dict[x]), len(user_countries))
service_country_one_hot = tf.one_hot(df['service_country'].apply(lambda x: service_country_dict[x]),
                                     len(service_countries))
user_as_one_hot = tf.one_hot(df['user_as'].apply(lambda x: user_as_dict[x]), len(user_as))
service_as_one_hot = tf.one_hot(df['service_as'].apply(lambda x: service_as_dict[x]), len(service_as))
print("OHE done")

# Define the embedding layer
embedding_size = 4
user_country_embedding = tf.keras.layers.Embedding(len(user_countries), embedding_size, input_length=1)
service_country_embedding = tf.keras.layers.Embedding(len(service_countries), embedding_size, input_length=1)
user_as_embedding = tf.keras.layers.Embedding(len(user_as), embedding_size, input_length=1)
service_as_embedding = tf.keras.layers.Embedding(len(service_as), embedding_size, input_length=1)
print("Embedding done")

# Compress each one-hot encoded feature using the corresponding embedding layer
user_country_compressed = user_country_embedding(user_country_one_hot)
service_country_compressed = service_country_embedding(service_country_one_hot)
user_as_compressed = user_as_embedding(user_as_one_hot)
service_as_compressed = service_as_embedding(service_as_one_hot)
print("Compressed")

# Flatten the compressed features
user_country_flat = tf.keras.layers.Flatten()(user_country_compressed)
service_country_flat = tf.keras.layers.Flatten()(service_country_compressed)
user_as_flat = tf.keras.layers.Flatten()(user_as_compressed)
service_as_flat = tf.keras.layers.Flatten()(service_as_compressed)

# Concatenate all flattened features
concatenated = tf.keras.layers.Concatenate()([user_country_flat, service_country_flat, user_as_flat, service_as_flat])
print("Concatenated")

# Define the DNN and CIN architectures using TensorFlow
input_shape = concatenated.shape[1]
output_shape = 1
print("Defined architecture")
# DNN
dnn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_shape)
])
print("DNN done")
# CIN
cin_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_shape)
])
print("CIN done")
# Define the final output layer for the combined model
combined_output = tf.keras.layers.Add()([dnn_model.output, cin_model.output])
final_output = tf.keras.layers.Dense(output_shape)(combined_output)
print("Final")
# Create the combined model
combined_model = tf.keras.models.Model(inputs=[dnn_model.input, cin_model.input], outputs=final_output)
print("Model combined")
# Train the combined model on the concatenated features as input
x = [concatenated, concatenated]
y = df['qos_response_time'].values

combined_model.compile(optimizer='adam', loss='mse')

combined_model.fit(x, y, epochs=100)
print("Model trained")


def evaluate_performance():
    # Evaluate performance of the model
    combined_loss = combined_model.evaluate(x, y)

    combined_rmse = np.sqrt(combined_loss)
    combined_mae = mean_absolute_error(y, combined_model.predict(x))
    print(f"Combined RMSE: {combined_rmse}, Combined MAE: {combined_mae}")


def predict_qos():
    # Perform one-hot encoding on each field in test data
    user_country_one_hot_test = tf.one_hot(test_df['user_country'].apply(lambda x: user_country_dict.get(x, -1)),
                                           len(user_countries))
    service_country_one_hot_test = tf.one_hot(
        test_df['service_country'].apply(lambda x: service_country_dict.get(x, -1)),
        len(service_countries))
    user_as_one_hot_test = tf.one_hot(test_df['user_as'].apply(lambda x: user_as_dict.get(x, -1)), len(user_as))
    service_as_one_hot_test = tf.one_hot(test_df['service_as'].apply(lambda x: service_as_dict.get(x, -1)),
                                         len(service_as))

    # Compress each one-hot encoded feature in test data using the corresponding embedding layer
    user_country_compressed_test = user_country_embedding(user_country_one_hot_test)
    service_country_compressed_test = service_country_embedding(service_country_one_hot_test)
    user_as_compressed_test = user_as_embedding(user_as_one_hot_test)
    service_as_compressed_test = service_as_embedding(service_as_one_hot_test)

    # Flatten the compressed features in test data
    user_country_flat_test = tf.keras.layers.Flatten()(user_country_compressed_test)
    service_country_flat_test = tf.keras.layers.Flatten()(service_country_compressed_test)
    user_as_flat_test = tf.keras.layers.Flatten()(user_as_compressed_test)
    service_as_flat_test = tf.keras.layers.Flatten()(service_as_compressed_test)

    # Concatenate all flattened features in test data
    concatenated_test = tf.keras.layers.Concatenate()(
        [user_country_flat_test, service_country_flat_test, user_as_flat_test, service_as_flat_test])

    # Use the trained model to make predictions on the test data
    predicted_qos = combined_model.predict([concatenated_test, concatenated_test])
    # add predicted qos values to dataframe
    test_df['predicted_qos'] = predicted_qos
    return test_df


def driver():
    evaluate_performance()
    result_df = predict_qos()
    print(tabulate(result_df.head(10), headers=result_df.columns, tablefmt='psql'))
    # Serialize the dataframe object
    with open('result_df.pickle', 'wb') as f:
        pickle.dump(result_df, f)
    print("Pickling done")


if __name__ == '__main__':
    driver()
