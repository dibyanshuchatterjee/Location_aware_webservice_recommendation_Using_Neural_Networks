import subprocess

train_and_predict_command = "python feature_interaction_learning_prediction.py 1000000"
get_recommendation_command = "python recommend.py 5"

# Run the commands and capture their output in real-time
train_and_predict_process = subprocess.Popen(train_and_predict_command, shell=True, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT)
get_recommendation_process = subprocess.Popen(get_recommendation_command, shell=True, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT)

# Print the output of each command as it runs
for process in [train_and_predict_process, get_recommendation_process]:
    for line in process.stdout:
        print(line.decode().strip())

# Wait for the commands to complete
train_and_predict_process.wait()
get_recommendation_process.wait()