import openai
import pandas as pd 
import csv
import os 
import random 
from datetime import datetime 
from dotenv import load_dotenv



# Setitng up the api call and key from env file
load_dotenv("OpenAI_key.env")
openai.organization = os.getenv("OPENAI_ORGANIZATION") 
openai.api_key = os.getenv("OPENAI_API_KEY")



# to run chat completions methods below make sure in your conda env openai==0.28 
def generate_sentences(system_prompt, n=100): # n specifies how many sentences are generated 
    sentences = []
    for _ in range(n):
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",  
            # for picking different models reference and their respective pricing check: https://openai.com/api/pricing/
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate a sentence based on the above instructions."}
            ],
            max_tokens=50
        )
        sentence = response.choices[0].message['content'].strip() # cleans generated sentences 
        sentences.append(sentence)
    return sentences

# reads and returns file as a string 
def load_system_prompt(filename):
    with open(filename, 'r') as file:
        return file.read()

# loads system prompt from local directory. System prompt specifies how to construct / measure oppositional intensity and directness 
system_prompt = load_system_prompt('system_prompt_level0.txt')



# Define the categories
categories = {
    "Low Directness Low Oppositional Intensity": ("Low", "Low"),
    "High Directness Low Oppositional Intensity": ("High", "Low"),
    "Low Directness High Oppositional Intensity": ("Low", "High"),
    "High Directness High Oppositional Intensity": ("High", "High")
}

# directory to save the generated CSV file
output_directory = '/Users/evanrowbotham/Dev/team-process-map/data/synthetic_data/synthetic_datasets/'

# confirm that the directory exists
os.makedirs(output_directory, exist_ok=True)

# joining path to csv file 
output_file_path = os.path.join(output_directory, 'generated_sentences.csv')



# Generate sentences and save to CSV
conversation_num = 1 
with open('generated_sentences.csv', 'w', newline='') as csvfile:
    fieldnames = ['conversation_num', 'speaker_id', 'message', 'timestamp', 'directness', 'oppositional_intensity'] 
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for description, (directness, oppositional_intensity) in categories.items():
        print(f"\nGenerating sentences for: {description}\n")
        
        # Custom prompt for each category
        category_prompt = system_prompt + f"\n\nPlease generate sentences with {directness} directness and {oppositional_intensity} oppositional intensity."
        
        sentences = generate_sentences(category_prompt)
        for idx, sentence in enumerate(sentences):
            writer.writerow({
                'conversation_num' : conversation_num, 
                'speaker_id' :  f"Speaker_{random.randint(0,5)}",
                'message' : sentence,
                'timestamp' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'directness' : directness,
                'oppositional_intensity' : oppositional_intensity

            })
            conversation_num += 1 

        print(f"Sentences for {description} saved to CSV.")

