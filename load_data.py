import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
import chardet


def read_data(filenames):
    df_list = list()
    for filename in filenames:
        print(os.getcwd())
        file_path = os.getcwd() + "/dataset1/" + filename

        # detect the encoding of the file
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']

        # read the file using the detected encoding # , error_bad_lines=False
        df = pd.read_csv(file_path, sep="\t", header=0, skiprows=[1], encoding=encoding)
        df_list.append(df)
    return df_list


def prepare_data(user_data_df, service_data_df):
    print("stage 1")
    user_data_cols = ["user_id", "ip_address", "user_country", "user_ip_no", "user_as", "user_latitude",
                      "user_longitude"]
    service_data_cols = ["service_id", "wsdl_address", "service_provider", "service_ip_address", "service_country",
                         "service_ip_no", "service_as", "service_latitude", "service_longitude"]
    user_service_cols = ["user_id", "service_id", "time_slice", "response_time"]
    file_path = "file://" + os.getcwd() + "/dataset2/" + "rtdata.txt"
    user_service_df = pd.read_csv(file_path, sep=" ", header=None, names=user_service_cols, skiprows=1)
    print("stage 2")
    user_data_df.columns = user_data_cols
    service_data_df.columns = service_data_cols
    user_service_df = user_service_df[user_service_cols]
    print(user_service_df.head(3))

    user_service_df = user_service_df.drop(columns=["time_slice"])
    user_data_df = user_data_df.drop(columns=["ip_address", "user_ip_no", "user_latitude", "user_longitude"])
    service_data_df = service_data_df.drop(
        columns=["wsdl_address", "service_provider", "service_ip_address", "service_ip_no", "service_latitude",
                 "service_longitude"])

    # perform joins:
    user_merged = pd.merge(user_service_df, user_data_df, on='user_id', how='left')
    service_merged = pd.merge(user_merged, service_data_df, on='service_id', how='left')
    print(service_merged.head(3))
    # Extract the AS number from the user_as column
    service_merged["user_as"] = service_merged["user_as"].str.extract("(AS\d+)", expand=False)
    # Extract the AS number from the service_as column
    service_merged["service_as"] = service_merged["service_as"].str.extract("(AS\d+)", expand=False)
    service_merged = service_merged.reindex(
        columns=['user_id', 'service_id', 'user_country', 'user_as', 'service_country', 'service_as',
                 'response_time']).rename(columns={'response_time': 'qos_response_time'})
    print("Dekho")
    print(tabulate(service_merged.head(3), headers=service_merged.columns, tablefmt='psql'))
    print("length: " + str(service_merged.shape[0]))

    return service_merged


def perform_encoding_and_transformation(data):
    # Extract relevant fields
    user_fields = ['user_id', 'user_country', 'user_as']
    service_fields = ['service_id', 'service_country', 'service_as']

    # Encode categorical features
    encoder = LabelEncoder()
    for field in user_fields + service_fields:
        data[field] = encoder.fit_transform(data[field])
    # transform_data(df=data)
    return data


def random_value():
    # 20% chance of skipping the cell
    if random.random() < 0.2:
        return None
    # generate a random integer between 0 and 5
    return random.randint(0, 5)


def assign_dummy_qos(pandas_df):
    # use apply method to add a new column 'qos_value' with random values to the DataFrame
    pandas_df['qos_value'] = pandas_df.apply(lambda row: random.randint(0, 5) if random.random() > 0.2 else np.nan,
                                             axis=1)

    print(pandas_df.head(10))
    return pandas_df


def driver():
    filenames = ["userlist.txt", "wslist.txt"]
    data = read_data(filenames=filenames)
    print("entering")
    data = prepare_data(data[0], data[1])
    return data


if __name__ == '__main__':
    driver()
