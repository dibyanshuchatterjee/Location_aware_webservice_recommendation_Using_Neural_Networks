# Install the requirements:
pip3 install -r requirements.txt

# Start with running feature_interaction_learning_prediction.py with an integer input defining how many rows to train the DNN and CIN with:
python feature_interaction_learning_prediction.py 1000000  

# Run recommend.py for final recommendation results with a top 'k' service argument:
python recommend.py 5  

# Run main.py to run the whole project at once
python main.py
