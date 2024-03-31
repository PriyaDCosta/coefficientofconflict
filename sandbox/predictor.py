from conflict_utils import *
from stage1_models import *
from stage2_models import *

import os

class PredictionBuilder():
    def __init__(self,model_1:str,model_2:str) -> None:
        self.model_1 = model_1
        self.model_2 = model_2

    """ 
    Checks if the pickle file for the input data exists, 
    else creates a pickle, and returns the input data
    
    """
    def get_pickled_inputs(self,input_pickle):

        if os.path.exists(input_pickle):
            return unpickle_embeddings(input_pickle)
        else:
            #Run the .ipynb notebook
            print("Generating all inputs, this may take upto 20 minutes.......")

            import papermill as pm
            pm.execute_notebook('preproces.ipynb')
        
    """ 
    Get the input data, based on the method specified for model 1 i.e. predicting 
    (1) Directness - Content, 
    (2) Directness - Expression,
    (3) Oppositional Intensity - Content, 
    (4) Oppositional Intensity - Expression

    based on the method specified in the input
    """
    def get_input_data(self):

        match self.model_1.lower():

            case "bert": #DONE

                train = self.get_pickled_inputs("embeddings/embeddings_hand_labeled.pickle") #Hand labeled data
                test = self.get_pickled_inputs("embeddings/embeddings_not_hand_labeled.pickle") 

                return classify_using_embeddings(train,test)
            
            case "chat_gpt": #TODO

                train = self.get_pickled_inputs("embeddings/chat_gpt_hand_labeled.pickle") #Hand labeled data
                test = self.get_pickled_inputs("embeddings/chat_gpt_not_hand_labeled.pickle") 

                return classify_using_chat_gpt(train,test)

            case "tpm": #DONE

                train = self.get_pickled_inputs("embeddings/tpm_hand_labeled.pickle") #Hand labeled data
                test = self.get_pickled_inputs("embeddings/tpm_not_hand_labeled.pickle") 

                return classify_using_tpm(train,test)

            
    """ 
    Based on 4 features, i.e
    (1) Directness - Content, 
    (2) Directness - Expression,
    (3) Oppositional Intensity - Content, 
    (4) Oppositional Intensity - Expression

    predict whether a conversation will go awry or will be a winning conversation
    """

    def get_predictions(self,input_data):
         
         match self.model_2.lower():

            case "lstm": #DONE
                
                return predict_using_lstm(input_data)

            case "neural_network": #DONE

                return predict_using_neural_net(input_data)
            
            case "logistic_regression": #DONE

                return predict_using_logistic_regression(input_data)
             
    """ 
    Get the model used to make the predictions for model 2  i.e. predicting whether the conversation 
    goes awry or becomes a winning conversation based on Directness (Content and Expression) and
    Oppositional Intensity (Content and Expression)
    """

    def predict(self):

        #Input data as per the method specified for Model 1
        input_data = self.get_input_data()
        print("Stage 1: Received Input Data as per " + str(self.model_1))
        
        #Predictions as per the method specified in Model 2
        actuals,predictions = self.get_predictions(input_data=input_data)
        print("Stage 2: Predicting as per " + str(self.model_2))

        #print metrics
        calculate_classification_metrics_per_class(actuals,predictions)
       
        