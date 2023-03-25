# The dataset used for the project is WS-Dream, collected by Zheng et al. Download the dataset from the below link:
https://github.com/wsdream/AMF/tree/master/data

# The work done in the project was derived from the publication: Location-Aware Feature Interaction Learning for Web Service Recommendation

# Install the requirements:
pip3 install -r requirements.txt

# Start with running feature_interaction_learning_prediction.py with an integer input defining how many rows to train the DNN and CIN with:
python feature_interaction_learning_prediction.py 1000000  

# Run recommend.py for final recommendation results with a top 'k' service argument:
python recommend.py 5  

# Run main.py to run the whole project at once
python main.py
