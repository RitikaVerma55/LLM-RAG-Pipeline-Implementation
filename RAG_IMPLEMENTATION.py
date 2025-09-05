!pip install langchain
!pip install pinecone
!pip install sentence_transformers
!pip install pinecone-client
import re
import pandas as pd
from io import BytesIO
from typing import Tuple, List
import pickle
import csv
import os
import json


import pinecone
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings


from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
product_data = pd.read_csv('/content/drive/MyDrive/products.csv')
product_data.head(2)
# converting csv to json

csv_file_path = '/content/drive/MyDrive/products.csv'
json_file_path = '/content/drive/MyDrive/products_data.json'

if not os.path.exists(json_file_path):
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = []

        for row in reader:
            data_object = {
                "Name": row["Name"],
                "Variant Name": row["Variant Name"],
                "Price": row["Price"],
                "Size": row["Size"],
                "Material": row["Material"],
                "Compatibility": row["Compatibility"],
                "Features": row["Features"],
                "URL": row["URL"],
                "Meta Title": row["Meta Title"],
                "Meta Description": row["Meta Description"],
                "Product Description": row["Product Description"]
            }

            data.append(data_object)

    json_data = json.dumps(data, indent=4)

    with open(json_file_path, 'w') as jsonfile:
        jsonfile.write(json_data)
else:
    print("JSON file already exists.")
# Splitting json file

def split_json_to_files(json_file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_file_path, 'r') as json_file:
        products = json.load(json_file)


    for i, product in enumerate(products):
        product_details = {
            'id': i,
            "Name": product["Name"],
            "Variant Name": product["Variant Name"],
            "Price": product["Price"],
            "Size": product["Size"],
            "Material": product["Material"],
            "Compatibility": product["Compatibility"],
            "Features": product["Features"],
            "URL": product["URL"],
            "Meta Title": product["Meta Title"],
            "Meta Description": product["Meta Description"],
            "Product Description": product["Product Description"]
        }

        output_file_path = os.path.join(output_dir, f'{i}.json')
        with open(output_file_path, 'w') as output_file:
            json.dump(product_details, output_file, indent=4)

        #print(f"Saved details of product {i} to {output_file_path}")

output_dir = '/content/drive/MyDrive/Prod_one_each'
json_file_path = '/content/drive/MyDrive/products_data.json'
split_json_to_files(json_file_path, output_dir)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# Get embeddings

def embed_product_information(json_file_path):
    with open(json_file_path, 'r') as json_file:
        product_data = json.load(json_file)

    text_attributes = [
        product_data['Name'],
        product_data['Variant Name'],
        product_data['Price'],
        ', '.join(eval(product_data['Size'])) if product_data['Size'] != 'NA' else '',
        ', '.join(eval(product_data['Material'])) if product_data['Material'] != 'NA' else '',
        ', '.join(eval(product_data['Compatibility'])) if product_data['Compatibility'] != 'NA' else '',
        ', '.join(eval(product_data['Features'])) if product_data['Features'] != 'NA' else '',
            product_data['Meta Title'],
        product_data['Meta Description'],
        product_data['Product Description']
    ]
    concatenated_text = '\n'.join(text_attributes)
    concatenated_text_with_image_url = concatenated_text + '\n' + product_data['URL']

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings_model = SentenceTransformer(model_name)
    embedding = embeddings_model.encode(concatenated_text_with_image_url )

    return embedding

# # Example usage
# json_file_path = '/content/drive/MyDrive/Prod_one_each/0.json'  # Path to your JSON file
# #embedding = embed_product_information(json_file_path)
# #print("Embedding:", embedding)
# embed = embed_product_information(json_file_path)
# print(embed)
def process_json_files(directory):
    embeddings_data = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(directory, filename)
            embedding = embed_product_information(json_file_path)

            with open(json_file_path, 'r') as json_file:
              product_data = json.load(json_file)

            product_id = product_data['id']
            product_name = product_data['Name']
            embeddings_data.append({'id': product_id, 'name': product_name, 'embedding': embedding.tolist()})

            # Save embedding data to a separate JSON file for each product
            output_json_path = os.path.join('/content/drive/MyDrive/embeddings', f'embedding_{product_id}.json')
            with open(output_json_path, 'w') as output_json_file:
                json.dump({'id': product_id, 'name': product_name, 'embedding': embedding.tolist()}, output_json_file)

    return embeddings_data


# Directory containing JSON files
directory_path = '/content/drive/MyDrive/Prod_one_each'

# Process JSON files in the directory
embeddings_data = process_json_files(directory_path)

# Save embeddings data to a JSON file
# with open('all_embeddings.json', 'w') as all_embeddings_file:
#     json.dump(embeddings_data, all_embeddings_file)
with open('all_embeddings.json', 'w') as all_embeddings_file:
    json.dump(embeddings_data, all_embeddings_file)
json_embed_path = "all_embeddings.json"

with open(json_embed_path, 'r') as json_file:
    embedded_product_data = json.load(json_file)

# Find the maximum length of a single embedding
max_embedding_length = max(len(embed["embedding"]) for embed in embedded_product_data)

print("Maximum length of a single embedding:", max_embedding_length)

#Connect with pinecone

pc = Pinecone(
        api_key='0ea3ebe6-7fc2-4312-9033-2f979891fd7e'
    )

if 'productdb' not in pc.list_indexes().names():
        pc.create_index(
            name='productdb',
            dimension= 384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
from pinecone import Index
#upserting into the database
# Set your index name (create it in Pinecone beforehand)
index_name = "productdb"


def store_product_embeddings(embeddings_dir):
    index = Index(api_key='0ea3ebe6-7fc2-4312-9033-2f979891fd7e', host ='https://productdb-hwxvhuf.svc.aped-4627-b74a.pinecone.io')

    for filename in os.listdir(embeddings_dir):
        if not filename.endswith(".json"):
            continue  # Ignore non-JSON files

        # Load product information
        with open(os.path.join(embeddings_dir, filename), "r") as f:
            product_data = json.load(f)

        # Extract product ID, name, and embedding from JSON data
        product_id = product_data.get("id")  # Use get() for safer access
        #product_name = product_data.get("name")  # Use get() for safer access
        embedding = product_data.get("embedding")  # Use get() for safer access

        if not all([product_id, embedding]):  # Check for missing data
            print(f"Warning: Skipping {filename} due to missing data.")
            continue

        # Combine product information and embedding into a single dictionary
        vector_data = {

                "id": str(product_id),  # Use product ID as embedding ID (optional)
                "values": embedding,
        }

        # Upsert data into Pinecone (using product ID as primary identifier)
        index.upsert([vector_data], index_name)

    print("Product information and embeddings stored successfully!")

# Example usage
embeddings_dir = "/content/drive/MyDrive/embeddings"  # Update with your path
store_product_embeddings(embeddings_dir)
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings_model = SentenceTransformer(model_name)
query = "apple phone cover"
query_embeddings = embeddings_model.encode(query)
query_embeddings = query_embeddings.tolist()
print(query_embeddings)
index = Index(api_key='0ea3ebe6-7fc2-4312-9033-2f979891fd7e', host ='https://productdb-hwxvhuf.svc.aped-4627-b74a.pinecone.io')

!pip install --upgrade pinecone
results = index.query(
    index_name='productdb',
    vector= query_embeddings,
    top_k = 10
)
print(results)
results= index.query(
    namespace="productdb",
    vector= query_embeddings,
    top_k=10,
    include_values=True,
    include_metadata=True,
    #filter={"genre": {"$eq": "action"}}
)
for result in results["matches"]:
    product_id = result["id"]  # Assuming "id" field exists in embedding
    similarity_score = result["score"]
    print(f"Product ID: {product_id}, Similarity Score: {similarity_score}")
