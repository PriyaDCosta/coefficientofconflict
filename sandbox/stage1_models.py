from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier

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

def classify_using_embeddings(train,test):

    # Apply the conversion function
    train['embeddings'] = train['embeddings'].apply(convert_to_array)
    test['embeddings'] = test['embeddings'].apply(convert_to_array)

    # Binary conversion of the target variables
    targets = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    train[targets] = (train[targets] > 0.5).astype(int)
    test[targets] = (test[targets] > 0.5).astype(int)

    # Prepare the features and labels
    X_train = np.stack(train['embeddings'].values)
    y_train = train[targets]

    X_test = np.stack(test['embeddings'].values)
    y_test = test[targets]

    # Use Logistic Regression for multi-label classification
    classifier = MultiOutputClassifier(LogisticRegression())

    # Train the model
    classifier.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = classifier.predict(X_test)

    # Calculate and print precision, recall, and F1 score for each target
    for i, target in enumerate(targets):
        precision = precision_score(y_test[target], y_pred[:, i])
        recall = recall_score(y_test[target], y_pred[:, i])
        f1 = f1_score(y_test[target], y_pred[:, i])

        print(f"{target}: Precision: {precision}, Recall: {recall}, F1: {f1}")

    # Append the predictions to the DF
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    pred_df = pd.concat([y_pred,y_pred_df], axis=1)

    # Reorder the columns
    pred_df = pred_df[train.columns]
        
    # Concatenate the train and test DF
    return pd.concat([train, pred_df], axis=0, ignore_index=True)



"""
For Classification using TPM Features
"""
def classify_using_tpm(train,test):
    features = [
    'num_words',
    'num_chars',
    'num_messages',
    'info_exchange_zscore_chats',
    'info_exchange_zscore_conversation',
    'discrepancies_lexical_per_100',
    'hear_lexical_per_100',
    'home_lexical_per_100',
    'conjunction_lexical_per_100',
    'certainty_lexical_per_100',
    'inclusive_lexical_per_100',
    'bio_lexical_per_100',
    'achievement_lexical_per_100',
    'adverbs_lexical_per_100',
    'anxiety_lexical_per_100',
    'third_person_lexical_per_100',
    'negation_lexical_per_100',
    'swear_lexical_per_100',
    'death_lexical_per_100',
    'health_lexical_per_100',
    'see_lexical_per_100',
    'body_lexical_per_100',
    'family_lexical_per_100',
    'negative_affect_lexical_per_100',
    'quantifier_lexical_per_100',
    'positive_affect_lexical_per_100',
    'insight_lexical_per_100',
    'humans_lexical_per_100',
    'present_tense_lexical_per_100',
    'future_tense_lexical_per_100',
    'past_tense_lexical_per_100',
    'relative_lexical_per_100',
    'sexual_lexical_per_100',
    'inhibition_lexical_per_100',
    'sadness_lexical_per_100',
    'social_lexical_per_100',
    'indefinite_pronoun_lexical_per_100',
    'religion_lexical_per_100',
    'work_lexical_per_100',
    'money_lexical_per_100',
    'causation_lexical_per_100',
    'anger_lexical_per_100',
    'first_person_singular_lexical_per_100',
    'feel_lexical_per_100',
    'tentativeness_lexical_per_100',
    'exclusive_lexical_per_100',
    'verbs_lexical_per_100',
    'friends_lexical_per_100',
    'article_lexical_per_100',
    'argue_lexical_per_100',
    'auxiliary_verbs_lexical_per_100',
    'cognitive_mech_lexical_per_100',
    'preposition_lexical_per_100',
    'first_person_plural_lexical_per_100',
    'percept_lexical_per_100',
    'second_person_lexical_per_100',
    'positive_words_lexical_per_100',
    'first_person_lexical_per_100',
    'nltk_english_stopwords_lexical_per_100',
    'hedge_words_lexical_per_100',
    'num_question_naive',
    'NTRI',
    'word_TTR',
    'first_pronouns_proportion',
    'function_word_accommodation',
    'content_word_accommodation',
    'mimicry_bert',
    'moving_mimicry',
    'hedge_naive',
    'textblob_subjectivity',
    'textblob_polarity',
    'dale_chall_score',
    'please',
    'please_start',
    'hashedge',
    'indirect_btw',
    'hedges',
    'factuality',
    'deference',
    'gratitude',
    'apologizing',
    '1st_person_pl',
    '1st_person',
    '1st_person_start',
    '2nd_person',
    '2nd_person_start',
    'indirect_greeting',
    'direct_question',
    'direct_start',
    'haspositive',
    'hasnegative',
    'subjunctive',
    'indicative',
    'forward_flow',
    'certainty_rocklage']

    # Binary conversion of the target variables
    targets = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    train[targets] = (train[targets] > 0.5).astype(int)
    test[targets] = (test[targets] > 0.5).astype(int)

    # Prepare the features and labels
    X_train = np.stack(train[features].values)
    y_train = train[targets]

    X_test = np.stack(test[features].values)
    y_test = test[targets]

    # Use Logistic Regression for multi-label classification
    classifier = MultiOutputClassifier(LogisticRegression())

    # Train the model
    classifier.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = classifier.predict(X_test)

    # Calculate and print precision, recall, and F1 score for each target
    for i, target in enumerate(targets):
        precision = precision_score(y_test[target], y_pred[:, i])
        recall = recall_score(y_test[target], y_pred[:, i])
        f1 = f1_score(y_test[target], y_pred[:, i])

        print(f"{target}: Precision: {precision}, Recall: {recall}, F1: {f1}")

    # Append the predictions to the DF
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    pred_df = pd.concat([y_pred,y_pred_df], axis=1)

    # Reorder the columns
    pred_df = pred_df[train.columns]
        
    # Concatenate the train and test DF
    return pd.concat([train, pred_df], axis=0, ignore_index=True)

""" 
Prediction using Chat GPT
"""
import pandas as pd
import fitz 
import openai

def classify_using_chat_gpt(train,test):

    # Initialize OpenAI with the API key
    openai.api_key = "API_KEY"
    
    # Read the PDF prompt
    with fitz.open("prompt.pdf") as doc:
        pdf_text = " ".join(page.get_text() for page in doc)

    additional_prompt = "Now Read the following examples.These are the 4 questions to answer"
    + "(1) Is the content of this text direct "
    + "(2) Is the experession of this chat direct?" 
    + "(3) Is this text oppositionally intense with respect to its content?"
    + "(4) Is this text oppositionally intense with respect to its expression?"
    + "For each row, label as 0 if the answer is no and 1 if the answer is yes for each. DO NOT JUSTIFY!"
    + "The labels for the 4 questions should be in 4 columns"
    + "i.e 'd_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average'"
    
    # Example of constructing a prompt from PDF text and potentially some insights from the training data
    prompt_base = pdf_text + additional_prompt

    # Iterate over the test data, generate a prompt for each row, and request a label from GPT-4
    labels = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    for index, row in test.iterrows():
        prompt = prompt_base + "how would you label this data: " + str(row.to_dict())
        response = openai.Completion.create(engine="text-davinci-004", prompt=prompt, max_tokens=50)
        
        # This part is highly dependent on how you format your prompt and the expected response format
        test.at[index, labels] = response.choices[0].text.strip().split()  # Example adjustment might be needed
    
    
    # Reorder the columns
    test = test[train.columns]
        
    # Concatenate the train and test DF
    return pd.concat([train, test], axis=0, ignore_index=True)