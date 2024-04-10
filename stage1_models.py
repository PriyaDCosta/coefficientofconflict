from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os
from key import *
from sklearn.multioutput import MultiOutputClassifier
from conflict_utils import *
import torch
from torch import nn
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

"""
For Classification using Embeddings
"""
# Adjusted function to handle different data types
def convert_to_array(embedding):
    if isinstance(embedding, str):  # Check if the embedding is a string
        # This is the case where embeddings are stored as string representations
        numbers = np.fromstring(embedding[1:-1], sep=' ')  # Assuming the format is '[n1 n2 n3 ...]'
        return numbers
    elif isinstance(embedding, (list, np.ndarray)):
        # If the embedding is already in the correct format (list or numpy array)
        return np.array(embedding)
    else:
        # Handle other types as needed
        raise ValueError(f"Unexpected data type: {type(embedding)}")

def classify_using_embeddings_log_reg(train,test):

    # Check if the pickle already exists : if it exists, just unpickle and return the output
    if os.path.exists("../sandbox/embeddings/stage_1_outputs/stage1_embeddings_output.pickle"):
       return unpickle_embeddings("../sandbox/embeddings/stage_1_outputs/stage1_embeddings_output.pickle")
    
    # Apply the conversion function
    train['embeddings'] = train['embeddings'].apply(convert_to_array)
    test['embeddings'] = test['embeddings'].apply(convert_to_array)

    # Binary conversion of the target variables
    targets = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    train[targets] = (train[targets] > 0.5).astype(int)

    # Prepare the features and labels
    X_train = np.stack(train['embeddings'].values)
    y_train = train[targets]

    X_test = np.stack(test['embeddings'].values)

    # Use Logistic Regression for multi-label classification
    classifier = MultiOutputClassifier(LogisticRegression())

    # Train the model
    classifier.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = classifier.predict(X_test)

    # Append the predictions to the DF
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    pred_df = pd.concat([test,y_pred_df], axis=1)

    for i, est in enumerate(classifier.estimators_):
        print(f"Label {i} feature weights: {est.coef_}")

    # Reorder the columns
    pred_df = pred_df[train.columns]
    
    # Store as a pickle
    pickle_embeddings(pred_df,"../sandbox/embeddings/stage_1_outputs/stage1_embeddings_output.pickle")
        
    # Concatenate the train and test DF
    return pred_df

def classify_using_reduced_embeddings_log_reg(train,test):

    # Check if the pickle already exists : if it exists, just unpickle and return the output
    if os.path.exists("../sandbox/embeddings/stage_1_outputs/stage1_embeddings_output.pickle"):
       return unpickle_embeddings("../sandbox/embeddings/stage_1_outputs/stage1_embeddings_output.pickle")
    
    # Reduce the embeddings
    train = reduce_embeddings_dimensionality(train, n_components=10)
    test = reduce_embeddings_dimensionality(test, n_components=10)

    # Apply the conversion function
    train['reduced_embeddings'] = train['reduced_embeddings'].apply(convert_to_array)
    test['reduced_embeddings'] = test['reduced_embeddings'].apply(convert_to_array)

    # Binary conversion of the target variables
    targets = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    train[targets] = (train[targets] > 0.5).astype(int)

    # Prepare the features and labels
    X_train = np.stack(train['reduced_embeddings'].values)
    y_train = train[targets]

    X_test = np.stack(test['reduced_embeddings'].values)

    # Use Logistic Regression for multi-label classification
    classifier = MultiOutputClassifier(LogisticRegression())

    # Train the model
    classifier.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = classifier.predict(X_test)

    # Append the predictions to the DF
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    pred_df = pd.concat([test,y_pred_df], axis=1)

    for i, est in enumerate(classifier.estimators_):
        print(f"Label {i} feature weights: {est.coef_}")

    # Reorder the columns
    pred_df = pred_df[train.columns]
    
    # Store as a pickle
    pickle_embeddings(pred_df,"../sandbox/embeddings/stage_1_outputs/stage1_embeddings_output.pickle")
        
    # Concatenate the train and test DF
    return pred_df


"""
For Classification using TPM Features
"""
def classify_using_tpm_with_logreg(train,test,features=ALL_FEATURES):
    # # If the output already exists as a pickle, just unpickle and share the output
    # if os.path.exists("../sandbox/embeddings/stage_1_outputs/stage1_tpm_output.pickle"):
    #    return unpickle_embeddings("../sandbox/embeddings/stage_1_outputs/stage1_tpm_output.pickle")
    
    #Replace all NaN with 0
    train.fillna(0.0, inplace=True)
    test.fillna(0.0, inplace=True)

    # Binary conversion of the target variables
    targets = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    train[targets] = (train[targets] > 0.5).astype(int)

    # Prepare the features and labels
    X_train = np.stack(train[features].values)
    y_train = train[targets]

    X_test = np.stack(test[features].values)

    # Use Logistic Regression for multi-label classification
    classifier = MultiOutputClassifier(LogisticRegression(class_weight="balanced"))

    # Train the model
    classifier.fit(X_train, y_train)
    
    # Predict on the testing set
    y_pred = classifier.predict(X_test)

    # Append the predictions to the DF
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    pred_df = pd.concat([test,y_pred_df], axis=1)

    for i, target in enumerate(targets):
        # Access the individual logistic regression model for the current target
        estimator = classifier.estimators_[i]
        
        # Retrieve the coefficients of the model
        coefficients = estimator.coef_[0]
        
        # Pair each feature name with its corresponding coefficient
        feature_importance = zip(features, coefficients)
        
        # Sort the features by the absolute value of their coefficient, in descending order
        sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
        
        # Select the top 5 most important features
        top_features = sorted_features[:5]
        
        # Print the top 5 predictive features and their coefficients for this target
        print(f"Top 5 predictive features for {target}:")
        for feature, coef in top_features:
            print(f"{feature}: {coef}")
        print("\n")

    # Store as a pickle
    pickle_embeddings(pred_df,"../sandbox/embeddings/stage_1_outputs/stage1_tpm_output.pickle")
        
    # Concatenate the train and test DF
    return pred_df

""" 
Prediction using Chat GPT
"""
import pandas as pd
import fitz 
import openai

def classify_using_chat_gpt(train,test):

    # If the output already exists as a pickle, just unpickle and share the output
    if os.path.exists("../sandbox/embeddings/stage_1_outputs/stage1_chatgpt_output.pickle"):
       return unpickle_embeddings("../sandbox/embeddings/stage_1_outputs/stage1_chatgpt_output.pickle")

    # Initialize OpenAI with the API key
    openai.api_key = API_KEY
    
    # Read the PDF prompt
    with fitz.open("prompt.pdf") as doc:
        pdf_text = " ".join(page.get_text() for page in doc)

    additional_prompt = (
        "Now Read the following examples.These are the 4 questions to answer "
        "(1) Is the content of this text direct "
        "(2) Is the expression of this chat direct? "
        "(3) Is this text oppositionally intense with respect to its content? "
        "(4) Is this text oppositionally intense with respect to its expression? "
        "For each row, label as 0 if the answer is no and 1 if the answer is yes for each. DO NOT JUSTIFY! "
        "The labels for the 4 questions should be in 4 columns "
        "i.e 'd_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average'"
    )
    
    # Example of constructing a prompt from PDF text and potentially some insights from the training data
    prompt_base = pdf_text + additional_prompt

    # Iterate over the test data, generate a prompt for each row, and request a label from GPT-4
    labels = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    for index, row in test.iterrows():
        prompt = prompt_base + "how would you label this data: " + str(row.to_dict())
        response = openai.Completion.create(engine="gpt-4", prompt=prompt, max_tokens=2)
        
        # This part is highly dependent on how you format your prompt and the expected response format
        test.at[index, labels] = response.choices[0].text.strip().split()  # Example adjustment might be needed
    
    
    # Reorder the columns
    test = test[train.columns]

    # Make predictions
    pred_df = pd.concat([train, test], axis=0, ignore_index=True)

    # Store as a pickle
    pickle_embeddings(pred_df,"../sandbox/embeddings/stage_1_outputs/stage1_chatgpt_output.pickle")
        
    # Concatenate the train and test DF
    return pred_df 

"""
For Classification using TPM Features
"""
def classify_using_tpm_reduced_dim(train,test,features=TOP_FEATURES):

    from sklearn.utils.class_weight import compute_class_weight

    # # If the output already exists as a pickle, just unpickle and share the output
    # if os.path.exists("../sandbox/embeddings/stage_1_outputs/stage1_tpm_output.pickle"):
    #    return unpickle_embeddings("../sandbox/embeddings/stage_1_outputs/stage1_tpm_output.pickle")
    
    #Replace all NaN with 0
    train.fillna(0.0, inplace=True)
    test.fillna(0.0, inplace=True)

    # Binary conversion of the target variables
    targets = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    train[targets] = (train[targets] > 0.5).astype(int)

    # Prepare the features and labels
    X_train = np.stack(train[features].values)
    y_train = train[targets]

    X_test = np.stack(test[features].values)

    # Use Logistic Regression for multi-label classification
    classifier = MultiOutputClassifier(LogisticRegression(class_weight="balanced"))

    # Train the model
    classifier.fit(X_train, y_train)
    
    # Predict on the testing set
    y_pred = classifier.predict(X_test)

    # Append the predictions to the DF
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    pred_df = pd.concat([test,y_pred_df], axis=1)

    for i, target in enumerate(targets):
        # Access the individual logistic regression model for the current target
        estimator = classifier.estimators_[i]
        
        # Retrieve the coefficients of the model
        coefficients = estimator.coef_[0]
        
        # Pair each feature name with its corresponding coefficient
        feature_importance = zip(features, coefficients)
        
        # Sort the features by the absolute value of their coefficient, in descending order
        sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
        
        # Select the top 5 most important features
        top_features = sorted_features[:5]
        
        # Print the top 5 predictive features and their coefficients for this target
        print(f"Top 5 predictive features for {target}:")
        for feature, coef in top_features:
            print(f"{feature}: {coef}")
        print("\n")

    # Store as a pickle
    pickle_embeddings(pred_df,"../sandbox/embeddings/stage_1_outputs/stage1_tpm_reduced_dim_output.pickle")
        
    # Concatenate the train and test DF
    return pred_df

def classify_using_tpm_with_xgboost(train, test, features):
    # Prepare data
    train.fillna(0.0, inplace=True)
    test.fillna(0.0, inplace=True)
    targets = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    train[targets] = (train[targets] > 0.5).astype(int)
    
    X_train = train[features]
    y_train = train[targets]
    X_test = test[features]
    
    # Define and train the model
    classifier = MultiOutputClassifier(xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'))
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    pred_df = pd.concat([test, y_pred_df], axis=1)

    # Feature importance - XGBoost provides a feature_importances_ attribute
    print("Feature importances:")
    for i, target in enumerate(targets):
        print(f"{target}:")
        importances = classifier.estimators_[i].feature_importances_
        for feat, importance in zip(features, importances):
            print(f"{feat}: {importance}")
        print("\n")
        
    return pred_df

class SimpleNN(nn.Module):
    def __init__(self, num_features):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(num_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 4)  # Assuming 4 targets
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

def classify_using_tpm_with_nn(train, test, features):
    # Prepare data
    scaler = StandardScaler()
    train.fillna(0.0, inplace=True)
    test.fillna(0.0, inplace=True)
    targets = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    train[targets] = (train[targets] > 0.5).astype(int)
    
    X_train_scaled = scaler.fit_transform(train[features])
    X_test_scaled = scaler.transform(test[features])
    y_train = train[targets].values
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    # Define the model
    model = SimpleNN(num_features=len(features))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    epochs = 100
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Make predictions
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_tensor = torch.sigmoid(model(X_test_tensor))  # Applying sigmoid to get probabilities
        y_pred_np = (y_pred_tensor.numpy() > 0.5).astype(int)  # Convert probabilities to binary output
    y_pred_df = pd.DataFrame(y_pred_np, columns=targets)
    pred_df = pd.concat([test.reset_index(drop=True), y_pred_df], axis=1)
    
    # Neural networks do not provide direct feature importance
    # but you can analyze weights or use techniques like SHAP for interpretation.
    
    return pred_df

def classify_using_tpm_with_decision_tree(train, test, features):
    # Define your target variables
    targets = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    
    # Fill missing values with 0.0 in both train and test sets
    train.fillna(0.0, inplace=True)
    test.fillna(0.0, inplace=True)

    # Binary conversion of the target variables
    train[targets] = (train[targets] > 0.5).astype(int)
    
    # Prepare the features and labels
    X_train = train[features]
    y_train = train[targets]
    X_test = test[features]
    
    # Define and initialize the Decision Tree classifier wrapped in a MultiOutputClassifier
    classifier = MultiOutputClassifier(DecisionTreeClassifier())
    
    # Train the model on the training data
    classifier.fit(X_train, y_train)
    
    # Predict on the testing set
    y_pred = classifier.predict(X_test)

    # Append the predictions to the test DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    pred_df = pd.concat([test, y_pred_df], axis=1)

    # Print feature importance for each target variable's model
    print("Feature Importances:")
    for i, target in enumerate(targets):
        print(f"Model for {target}:")
        importances = classifier.estimators_[i].feature_importances_
        for feat, importance in zip(features, importances):
            print(f"  {feat}: {importance}")
        print("\n")
        
    return pred_df




