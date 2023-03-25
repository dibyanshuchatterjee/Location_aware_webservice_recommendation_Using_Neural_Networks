import argparse
import pickle

# create argument parser object
parser = argparse.ArgumentParser()

# add required argument
parser.add_argument('top_k', help='Gives top k result for services')

# parse arguments
args = parser.parse_args()

# Load the dataframe object from the pickle file
with open('result_df.pickle', 'rb') as f:
    df = pickle.load(f)


# Select the top-k services with the least predicted QoS values
k = int(args.top_k)
best_services = (
    df.groupby('service_id')
    .agg({'qos_response_time': 'min'})
    .sort_values('qos_response_time')
    .head(k)
    .reset_index()['service_id']
    .tolist()
)

print("The top-{} best services (service-ids) to use with the least response time are:".format(k))
print(best_services)
