import pandas as pd
import os
import pickle
import logging
from datetime import datetime

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predefined_models.text_embedding import get_text_embedding_model



# Logging setup
log_file = './logs/preprocess_pipeline.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
data_path = './data/dialogue/dialogs.csv'
pickle_path = './data/dialogue/dialog_preprocessed.pkl'  
exception_log = './exceptions/preprocess_exceptions.log'

# Load the SentenceTransformer model
model = get_text_embedding_model()

# Function to preprocess text data
def preprocess_data(chunk_size=5000, batch_size=100):
    try:
        logging.info('Starting preprocessing')        
        df = pd.read_csv(data_path)
        
        # Drop duplicates based on 'text' column, ignoring 'dialog_id'
        df = df.drop_duplicates(keep='first').copy()
        
        # Add 'status' column with default value 'False'
        if 'status' not in df.columns:
            df['status'] = False

        # Load existing preprocessed data if available
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as file:
                preprocessed_data = pickle.load(file)
        else:
            preprocessed_data = pd.DataFrame(columns=df.columns.tolist() + ['vector'])

        # Get unprocessed rows
        unprocessed_rows = df[df['status'] == False]
        
        # Process in chunks
        initial_chunk = unprocessed_rows.iloc[:chunk_size]
        process_and_save(initial_chunk, preprocessed_data)

        # Process in incremental batches
        for start in range(chunk_size, len(unprocessed_rows), batch_size):
            batch = unprocessed_rows.iloc[start:start + batch_size]
            process_and_save(batch, preprocessed_data)

        logging.info('Preprocessing complete. Saved to pickle file.')
    except Exception as e:
        logging.error(f'Error during preprocessing: {e}')
        with open(exception_log, 'a') as ex_file:
            ex_file.write(f'{datetime.now()} - {str(e)}\n')

# Function to process a batch and save to pickle file
def process_and_save(batch, preprocessed_data):
    if batch.empty:
        return

    try:
        # Generate embeddings
        batch['vector'] = batch['text'].apply(lambda x: model.encode(x).tolist())
        batch['status'] = True

        # Update preprocessed data
        preprocessed_data = pd.concat([preprocessed_data, batch], ignore_index=True)
        
        # Save to pickle
        with open(pickle_path, 'wb') as file:
            pickle.dump(preprocessed_data, file)

        logging.info(f'Processed and saved {len(batch)} records.')
    except Exception as e:
        logging.error(f'Error processing batch: {e}')
        with open(exception_log, 'a') as ex_file:
            ex_file.write(f'{datetime.now()} - {str(e)}\n')

if __name__ == '__main__':
    preprocess_data()
#change this code that will store embedding data in Supabase 
# Add code for Stripping whitespace, Lowercasing, and removing stopwords

'''
#wrute preprocessed data to supabase 
from supabase import create_client, Client

# Supabase setup
SUPABASE_URL = ""
SUPABASE_KEY = ""
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def process_and_save(batch):
    if batch.empty:
        return

    try:
        # Preprocess text
        batch['text'] = batch['text'].str.strip().str.lower()
        
        # Generate embeddings
        batch['vector'] = batch['text'].apply(lambda x: model.encode(x).tolist())
        batch['status'] = True

        # Insert into Supabase
        data_to_insert = batch[['text', 'vector', 'status']].to_dict(orient='records')
        response = supabase.table("dialog_embeddings").insert(data_to_insert).execute()

        if response.get("error"):
            logging.error(f"Supabase error: {response['error']}")
        else:
            logging.info(f"Inserted {len(data_to_insert)} records into Supabase.")

    except Exception as e:
        logging.error(f'Error processing batch: {e}')'
        '''
