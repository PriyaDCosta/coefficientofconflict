import torch
import pickle
import re
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import precision_recall_fscore_support


def preprocess_for_attention(df):

    #if t
    df_col = 'CONV_ID'
    if 'conversation_num' in df.columns:
        df_col = 'conversation_num'
    
    # Example DataFrame creation (replace this with your actual DataFrame loading)
    np.random.seed(19104)  # For reproducible random results

    # Normalize features
    scaler = StandardScaler()
    features = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']
    df[features] = scaler.fit_transform(df[features])

    """
    Grouping by CONV_ID:

    This line groups the DataFrame df by the column CONV_ID. 
    Each group corresponds to a unique conversation identified by CONV_ID. 
    The purpose is to treat each conversation as a sequence, which is particularly useful for sequence modeling tasks where the context of the conversation is important. 
    """
    grouped = df.groupby(df_col)
    sequences = []
    targets = []

    """
    Prepare Sequences and Targets:

    Iterates over each group created by the groupby operation.

    seq = group[features].values extracts just the values of the specified features 
    (features is a list of column names) from each group as a NumPy array. 
    This array represents the sequence of observations for a single conversation.

    target = group['dataset_numeric'].values[0] extracts the target variable for the sequence. 
    This example takes the last value of the dataset_numeric column from the group as the target. 
    The assumption here might be that the target of the entire sequence (conversation) is determined by its final state or message. 
    """

    for _, group in grouped:
        seq = group[features].values  # Extract features as sequence
        target = group['dataset_numeric'].values[0]  # Extract the target variable for the sequence. All the values are same for a given CONV_ID i.e the 0 for awry or 1 for winning
        sequences.append(torch.tensor(seq, dtype=torch.float))
        targets.append(torch.tensor(target, dtype=torch.float))

    """
    Padding Sequences:

    Since the sequences (conversations) can have varying lengths (i.e., different numbers of messages or observations),
    they need to be padded to have the same length to be processed in batches by the model. 
    
    The pad_sequence function from PyTorch's torch.nn.utils.rnn module achieves this by adding zeros to shorter sequences until all sequences in the batch have the same length.

    The parameter batch_first=True indicates that the output tensor should have a batch size as its first dimension, i.e.,
    the tensor shape will be (batch_size, seq_length, features), which is the format expected by most PyTorch models for batched sequence data 
    """

    # Padding sequences to have the same length
    padded_sequences = pad_sequence(sequences, batch_first=True)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, targets, test_size=0.2, random_state=19104)

    # Convert lists to tensor for targets if necessary
    y_train = torch.stack(y_train)
    y_test = torch.stack(y_test)
    
    return X_train, X_test, y_train, y_test

    
def break_long_messages(df, threshold=200):
    new_rows = []
    
    for index, row in df.iterrows():
        text = row['text']
        message_id = row['id']
        

        if("Diversity means more people" in row["text"]):
            print(row)
            print(row["text"])
        """
        Check if: 
        - the message exceeds the threshold; 
        - there is a newline (\n) in the message we can split on
        - only do this if not OP
        """
        if len(text.split()) > threshold and '\n' in text and  row['reply_to'] != "ORIGINAL_POST":
            chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]

            current_chunk_id = 1
            current_chunk = ""
            gt = False
            for chunk_num, chunk in enumerate(chunks):
                current_chunk += chunk

                # SEPARATELY HANDLE QUOTES: if an earlier chunk contains '&gt;' as a character, this is quoting another person.
                # We want the quote and the response to be grouped together as a single 'idea.'
                # Detect the "&gt;" character and save it separately if this is the case.  
                if("&gt;" in chunk):
                    current_chunk += "\n"
                    gt = True # toggle 'gt' on if this is the case
                elif(gt and not "&gt;" in chunk): # We found the response to the quote, so save it.
                    new_row = row.copy()
                    new_row['text'] = current_chunk.strip()
                    new_row['id'] = f"{message_id}_{current_chunk_id}"
                    new_rows.append(new_row)
                    current_chunk_id += 1
                    current_chunk = ""
                    gt = False
                elif len(re.sub('\n', '', current_chunk).split()) > threshold: # chunk is long enough; save directly
                    new_row = row.copy()
                    new_row['text'] = current_chunk.strip()
                    new_row['id'] = f"{message_id}_{current_chunk_id}"
                    new_rows.append(new_row)
                    current_chunk_id += 1
                    current_chunk = ""
                else: # not long enough; add next chunk
                    current_chunk += "\n"
            if(current_chunk != ""):
                # add the last chunk to the df, if it wasn't long enough
                new_row = row.copy()
                new_row['text'] = current_chunk.strip()
                new_row['id'] = f"{message_id}_{current_chunk_id}"
                new_rows.append(new_row)
        else:
            new_rows.append(row) # No changes necessary
    
    # Create a new DataFrame with the updated rows
    new_df = pd.DataFrame(new_rows, columns=df.columns)
    
    return new_df

""" 
Merge the text to the ratings file
"""
def merge_raw_data(data,winning_df,awry_df):
    cols = ['CONV_ID','id', 'text','speaker','timestamp', 'meta.score','reply_to', 'conversation_length']

    # Drop duplicate rater ids
    data = data[data['rater_id'] == 'amy']
        
    # Add columns to denote the dataset i.e. winning or awry
    winning_df['dataset_numeric'] = 1
    awry_df['dataset_numeric'] = 0

    # Drop the OP posts
    winning_df = drop_op(winning_df)

    # Merge the complete datasets (winning + awry, labeled and non-labeled)
    merged_complete_data = pd.concat([winning_df, awry_df], ignore_index=True) 

    # Combine with the text data (this is winning + awry hand labeled)
    hand_labeled_data = pd.merge(data, merged_complete_data[cols], on=['CONV_ID', 'id'], how='left')
    hand_labeled_data['human_labels'] = 1

    # Get the list of hand labeled rows from the original dataset
    unique_hand_labeled_convo_ids  = hand_labeled_data['CONV_ID'].unique()

    # Separate non-hand labeled data from other data
    non_hand_labeled_data = merged_complete_data[~merged_complete_data['CONV_ID'].isin(unique_hand_labeled_convo_ids)]
    
    # # Convert the text to str (from obj)
    hand_labeled_data['text'] = hand_labeled_data['text'].astype(str)
    non_hand_labeled_data['text'] = non_hand_labeled_data['text'].astype(str)

    #Split the tokens
    hand_labeled_data = break_long_messages(hand_labeled_data)
    non_hand_labeled_data = break_long_messages(non_hand_labeled_data)

    print(hand_labeled_data.columns)
    print(non_hand_labeled_data.columns)

    return hand_labeled_data,non_hand_labeled_data


"""
Drop the OP's message for winning conversations
"""
def drop_op(df):

    # Check if there is any row with 'dataset' column value as 'winning'
    if (df['dataset_numeric'] == 1).any():
        # Find the first 'CONV_ID' for which 'dataset' column value is 'winning'
        first_winning_conv_id = df[df['dataset_numeric'] == 1]['CONV_ID'].iloc[0]
        
        # Drop the row(s) with this 'CONV_ID'
        df = df[df['CONV_ID'] != first_winning_conv_id]

    return df

"""
Drop the OP's message for winning conversations
"""
def drop_op_tpm(df):

    # Check if there is any row with 'dataset' column value as 'winning'
    if (df['dataset_numeric'] == 1).any():
        # Find the first 'CONV_ID' for which 'dataset' column value is 'winning'
        first_winning_conv_id = df[df['dataset_numeric'] == 1]['conversation_num'].iloc[0]
        
        # Drop the row(s) with this 'CONV_ID'
        df = df[df['conversation_num'] != first_winning_conv_id]

    return df


"""
Convert the labels into numeric scores
"""

def get_numeric_labels(text):

    # Convert the text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Initialize the result variable
    result = 0
    
    # Check if "yes" is present in the text
    if 'yes' in text_lower:
        result = 1
    elif 'no' in text_lower:
        result = 0
    
    return result

"""
Convert all the columns to numeric labels
"""
def convert_labels(df):
    
    df['d_content'] = df['rating_directness_content'].apply(get_numeric_labels)
    df['d_expression'] = df['rating_directness_expression'].apply(get_numeric_labels)
    df['oi_content'] = df['rating_OI_content'].apply(get_numeric_labels)
    df['oi_expression'] = df['rating_OI_expression'].apply(get_numeric_labels)
    df['dataset_numeric'] = df['dataset'].apply(get_dataset_numeric_labels)


"""
Get the average of the ratings for a single column
"""
def get_averages(df,on_column,conv_id='CONV_ID'):

    # Calculate average ratings
    average_ratings = df.groupby([conv_id, 'id'])[on_column].mean().reset_index()

    # Merge average ratings with original DataFrame
    df = df.merge(average_ratings, on=[conv_id, 'id'], how='left', suffixes=('', '_average'))

    return df


"""
Get the average ratings for all the columns
"""
def average_labels(df, columns,conv_id):
    for column in columns:
        df = get_averages(df, column,conv_id)
    return df


"""
Determine the labels for the dataset
"""
def get_label(conv_id):
    if conv_id.endswith('_A') or conv_id.endswith('_B'):
        return 'winning'
    else:
        return 'awry'

""" 
Get the dataset which the conversation belongs to awry or winning
"""
def dataset_labels(df,on_column='CONV_ID'):
    df['dataset'] = df[on_column].apply(get_label)
    

"""
Drop unncessary columns 
"""
def drop_cols(df,type):
    if type == 'average':
        return df[['d_content_average', 'd_expression_average', 'oi_content_average','oi_expression_average', 'dataset']]
    else:
        return df[['d_content', 'd_expression', 'oi_content','oi_expression','dataset']]
    
def get_dataset_numeric_labels(text):

    # Convert the text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Initialize the result variable
    result = 0
    
    # Check if "yes" is present in the text
    if 'winning' in text_lower:
        result = 1
    elif 'awry' in text_lower:
        result = 0
    
    return result


""" 
Format inputs for TPM features
"""

def format_for_tpm(df):

    df = df.rename(columns={'CONV_ID' : 'conversation_num', 'text':'message' ,'speaker': 'speaker_nickname'})
    df['message'] = df['message'].astype(str)
    df['message'] = df['message'].astype(str)

    # Drop largest Convos
    large_convos_list = large_convos(df)
    df = df[~df['conversation_num'].isin(large_convos_list)]
    
    # get_message_length(df)
    return df


"""
Pickle embeddings 
"""
def pickle_embeddings(input_file,output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(input_file, file)

"""
Unpickle embeddings 
"""
def unpickle_embeddings(input_file):
    with open(input_file, 'rb') as file:
        return pd.DataFrame(pickle.load(file))
    

"""
Generates SBERT embeddings for the text data in the specified column of a DataFrame and adds
these embeddings as a new column to the DataFrame.

Args:
- dataframe (pd.DataFrame): DataFrame containing the text data.
- text_column (str): Column name of the text data to convert to embeddings.

Returns:
- pd.DataFrame: Original DataFrame with a new column 'embeddings' containing SBERT embeddings.
    """

def generate_sbert_embeddings(dataframe, text_column,n_components=4):
    # Load the SBERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert the specified text column to a list
    texts = dataframe[text_column].tolist()
    
    # Generate embeddings
    embeddings = model.encode(texts, convert_to_tensor=False, batch_size=32, show_progress_bar=True)
    
    # Initialize PCA and fit to the embeddings to reduce to 4 dimensions
    pca = PCA(n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Assign reduced embeddings to a new column in the DataFrame
    dataframe['embeddings'] = list(reduced_embeddings)
    
    return dataframe

"""
Calculate precision, recall, and F1 score for each class in the given true and predicted values.

Args:
- y_true (array-like): True labels.
- y_pred (array-like): Predicted labels.

Returns:
- A dictionary containing the precision, recall, and F1 score for each class.
"""
def calculate_classification_metrics_per_class(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Create a dictionary for easier reading of results
    metrics = {
        'precision': precision,
        'recall': recall,
        'F1_score': f1,
    }

    print("Precision by class:  ", "Awry: ",round(metrics['precision'][0],2),"  ","Winning: ",round(metrics['precision'][1],2))
    print("Recall by class:     ", "Awry: ",round(metrics['recall'][0],2),"  ","Winning: ",round(metrics['precision'][1],2))
    print("F1 Score by class:   ", "Awry: ",round(metrics['F1_score'][0],2),"  ","Winning: ",round(metrics['precision'][1],2))

""" 
Conversation ID of text with more than 500 words
"""
def large_convos(df):

    filtered_df = df[df['message'].str.split().apply(len) > 200]
    conversation_ids = filtered_df['conversation_num'].unique()
    print(len(conversation_ids.tolist()))

    return conversation_ids.tolist()

def get_stage1_label_counts(df):

    columns = ['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average','dataset_numeric']

    # Initialize dictionaries to hold counts
    ones_counts = {}
    zeros_counts = {}

    total = len(df)

    for col in columns:
        # Count values for the current columns
        counts = df[col].value_counts()
        
        # Store the counts of 1's and 0's
        ones_counts[col] = round(counts.get(1, 0) / total,2)
        zeros_counts[col] = round(counts.get(0, 0) / total,2)

    # Display the counts
    print("Counts of 1's:")
    print(ones_counts)
    print("\nCounts of 0's:")
    print(zeros_counts)

import pandas as pd
import numpy as np
import umap

def reduce_embeddings_dimensionality(df, n_components=10):
    """
    Reduce the dimensionality of the 'embeddings' column in a DataFrame using UMAP.

    Parameters:
    - df: A pandas DataFrame containing an 'embeddings' column, where each row is an array-like embedding.
    - n_components: The number of dimensions to reduce the embeddings to (default is 2).

    Returns:
    - A new DataFrame with the original data and an additional column 'reduced_embeddings'
      containing the embeddings with reduced dimensionality.
    """
    # Extract embeddings into a list of arrays
    embeddings = list(df['embeddings'])

    # Ensure all embeddings are numpy arrays (assuming they're not already)
    embeddings = [np.array(embedding) for embedding in embeddings]
    
    # Stack the list of arrays into a single numpy array for UMAP
    embeddings_stack = np.vstack(embeddings)

    # Initialize and fit UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings_stack)

    # Store the reduced embeddings back into the DataFrame
    df['reduced_embeddings'] = list(reduced_embeddings)

    return df

def evaluate_model_with_multiple_metrics(model,X, y, n_splits=10):

    # Define the k-fold cross-validation procedure
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Define metrics to evaluate
    scoring = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    
    # Evaluate the model
    scores = cross_validate(model, X, y, scoring=scoring, cv=kf, return_train_score=False)
    
    # Calculate the average scores across all folds
    avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
    
    print(avg_scores)

import re

"""
Remove text that appears within quotes, specifically between two ">" characters.

Parameters:
- text: A string containing the original text.

Returns:
- The text with all quoted text between ">" characters removed.
"""
def remove_quotes(text):

    # Define the regex pattern to match text within ">"
    # Pattern explanation:
    # - &gt;: Matches the literal character ">"
    # - .*?: Matches any character (.) any number of times (*), as few times as possible (?)
    #        to ensure it matches the shortest sequence between ">"
    # - &gt;: Matches the literal closing ">"
    pattern = r"&gt;.*?&gt;"
    
    # Use re.sub() to replace all occurrences of the pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

def remove_quotes_from_dataset(df,on_column):

    df[on_column] = df[on_column].apply(remove_quotes)
