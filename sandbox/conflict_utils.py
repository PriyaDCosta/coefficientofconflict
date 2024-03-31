import torch
import pickle
import random
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support

def preprocess_for_attention(df):
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
    grouped = df.groupby('CONV_ID')
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

    return padded_sequences,targets
    


""" 
Merge the text to the ratings file
"""
def merge_raw_data(data,winning_df,awry_df):

    cols = ['CONV_ID', 'id', 'text','speaker','timestamp', 'meta.score','reply_to', 'conversation_length']
    
    merged_df = pd.merge(data, winning_df[cols], on=['CONV_ID', 'id'], how='left') 
    merged_df = pd.merge(merged_df, awry_df[cols], on=['CONV_ID', 'id'], how='left')

   # List of column pairs to merge: (source_column_1, source_column_2, target_column)
    columns_to_merge = [
        ('text_x', 'text_y', 'text'),
        ('speaker_x', 'speaker_y', 'speaker'),
        ('timestamp_x', 'timestamp_y', 'timestamp'),
        ('meta.score_x', 'meta.score_y', 'meta.score'),
        ('reply_to_x', 'reply_to_y', 'reply_to'),
        ('conversation_length_x', 'conversation_length_y', 'conversation_length'),
    ]

    # Iterate over the column pairs and merge them
    for col1, col2, new_col in columns_to_merge:
        merged_df[new_col] = np.where(pd.isna(merged_df[col1]), merged_df[col2], merged_df[col1])

    'text_x','speaker_x','timestamp_x', 'meta.score_x', 'reply_to_x','conversation_length_x',
    'text_y','speaker_y','timestamp_y', 'meta.score_y', 'reply_to_y','conversation_length_y',

    #Drop duplicate rater ids
    merged_df = merged_df[merged_df['rater_id'] == 'amy']

    #Drop unncessary columns
    merged_df = merged_df.drop(columns=['rating_directness_content','rating_directness_expression',
    'rating_OI_content','rating_OI_expression', 'rater_id', 'status', 'last_updated_time',
    'dataset', 'd_content', 'd_expression', 'oi_content', 'oi_expression','text_x','speaker_x','timestamp_x', 'meta.score_x', 'reply_to_x','conversation_length_x',
    'text_y','speaker_y','timestamp_y', 'meta.score_y', 'reply_to_y','conversation_length_y'])

    merged_df['human_labels'] = 1
    awry_df['human_labels'] = 0
    winning_df['human_labels'] = 0

    awry_df['dataset_numeric'] = 0
    winning_df['dataset_numeric'] = 1

    #drop hand labeled rows from the main datasets
    unique_convo_ids = merged_df['CONV_ID'].unique()

    #Combine non-hand labeled data 
    combined_df = pd.concat([awry_df, winning_df], ignore_index=True)
    filtered_combined_df = combined_df[~combined_df['CONV_ID'].isin(unique_convo_ids)]

    return filtered_combined_df,merged_df


"""
Drop the OP's message for winning conversations
"""
def drop_op(df):

    # Check if there is any row with 'dataset' column value as 'winning'
    if (df['dataset'] == 'winning').any():
        # Find the first 'CONV_ID' for which 'dataset' column value is 'winning'
        first_winning_conv_id = df[df['dataset'] == 'winning']['CONV_ID'].iloc[0]
        
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

    print("Precision by class:", metrics['precision'])
    print("Recall by class:", metrics['recall'])
    print("F1 Score by class:", metrics['F1_score'])