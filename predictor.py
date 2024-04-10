from conflict_utils import *
from stage1_models import *
from stage2_models import *

import os

class PredictionBuilder():
    def __init__(self,data_source:str,dimensions:str,regularization:str,stage_1_model:str,stage_2_model:str) -> None:
        self.data_source = data_source
        self.dimensions = dimensions
        self.regularization = regularization
        self.stage_1_model = stage_1_model
        self.stage_2_model = stage_2_model

    """ 
    Checks if the pickle file for the input data exists, 
    else creates a pickle, and returns the input data
    
    """
    def get_pickled_inputs(self,input_pickle):

        if os.path.exists(input_pickle):
            return unpickle_embeddings(input_pickle)
        else:
            #Run the .ipynb notebook
            print("Generating all inputs, this may take a long time if BERT Embeddings are being generated.......")

            import papermill as pm
            pm.execute_notebook('preprocess.ipynb')
        
    """ 
    Get the input data, based on the method specified for model 1 i.e. predicting 
    (1) Directness - Content, 
    (2) Directness - Expression,
    (3) Oppositional Intensity - Content, 
    (4) Oppositional Intensity - Expression

    based on the method specified in the input

    All the methods will return a single df with hand-labeled + non-hand-labeled data
    """
    def get_input_data(self):

        train_df = None
        test_df = None
        regularizer = None
        data_set = None

        # Get the data
        match self.data_source.lower():

            case "embeddings": 

                data_set = "embeddings"

                # Get the dimension
                match self.dimensions.lower():

                    case "reduced_dim": 
                        
                        train_df = self.get_pickled_inputs("embeddings/initial_inputs/embeddings_hand_labeled.pickle") #Hand labeled data
                        test_df = self.get_pickled_inputs("embeddings/initial_inputs/embeddings_non_hand_labeled.pickle") 

                    case "full_dim":

                        train_df = self.get_pickled_inputs("embeddings/initial_inputs/embeddings_hand_labeled.pickle") #Hand labeled data
                        test_df = self.get_pickled_inputs("embeddings/initial_inputs/embeddings_non_hand_labeled.pickle") 

            # Get the dimension 
            case "tpm":

                data_set = "tpm"
                
                match self.dimensions.lower():

                    case "reduced_dim": 
                        
                        train_df = self.get_pickled_inputs("embeddings/initial_inputs/tpm_hand_labeled.pickle") #Hand labeled data
                        test_df = self.get_pickled_inputs("embeddings/initial_inputs/tpm_non_hand_labeled.pickle") 

                    case "full_dim":

                        train_df = self.get_pickled_inputs("embeddings/initial_inputs/tpm_hand_labeled.pickle") #Hand labeled data
                        test_df = self.get_pickled_inputs("embeddings/initial_inputs/tpm_non_hand_labeled.pickle") 

        # Get the regularization
        match self.regularization.lower():

            case "l1": 
                regularizer = "l1"
                
            case "l2":
                regularizer = "l2"

            case "elastic_net":
                regularizer = "elastic_net"

        # Get the model

        match data_set:

            case "tpm": #TPM

                match self.stage_1_model.lower():
                    
                    case "logreg": #DONE

                        return classify_using_tpm_with_logreg(train_df,test_df,regularizer)
                    
                    case "decision_trees": #DONE

                        return classify_using_tpm_with_decision_tree(train_df,test_df,regularizer)

                    case "xgboost": #DONE

                        return classify_using_tpm_with_xgboost(train_df,test_df,regularizer)
                    
                    case "neural_network": #DONE

                        return classify_using_tpm_with_nn(train_df,test_df,regularizer)
                    
            case "embeddings":

                 match self.stage_1_model.lower():
                    
                    case "logreg": #DONE

                        return classify_using_embeddings_log_reg(train_df,test_df,regularizer)
                    
                    case "decision_trees": #TODO

                        pass

                    case "xgboost": #TODO

                        pass
                    
                    case "neural_network": #TODO

                        pass
                    
            
            
    """ 
    Based on 4 features, i.e
    (1) Directness - Content, 
    (2) Directness - Expression,
    (3) Oppositional Intensity - Content, 
    (4) Oppositional Intensity - Expression

    predict whether a conversation will go awry or will be a winning conversation
    """

    def get_predictions(self,input_data):
         
         match self.stage_2_model.lower():

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

        # Input data as per the method specified for Model 1
        input_data = self.get_input_data()
        print("Stage 1: Received Input Data as per " + str(self.model_1))
        print(" ")

        
        # Predictions as per the method specified in Model 2
        actuals,predictions = self.get_predictions(input_data=input_data)
        print("Stage 2: Predicting as per " + str(self.model_2))
        print(" ")

        # Print metrics
        calculate_classification_metrics_per_class(actuals,predictions)
       
        