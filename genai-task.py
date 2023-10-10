import logging
import requests
import json
import openai
import os
import sys

from langchain.llms import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader

import chromadb


# Set up environment variables for connection to azure OpenAI
openai.api_key = 'fd925220485248079613dfc0bbfea02b'
openai.api_base = 'https://ai-recommendations.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2023-05-15'

deployment_name = 'genai-embedder'


# Function to connect to Azure OpenAI and send a query
def send_query(query):
    print("test query")
    query = """Please translate the following question into a SQL query to be used on a dataset where the following DICT describes the dataset by 
    giving the column headers and a brief description."""
    print(query)
    response = openai.Completion.create(engine=deployment_name, prompt=query, max_tokens = 50)
    return response

# Response of above test query
'''
response below
{
  "id": "cmpl-87n0EvuNU8MxrTj8ZFBfdSOWf5FvN",
  "object": "text_completion",
  "created": 1696867354,
  "model": "gpt-35-turbo",
  "choices": [
    {
      "text": "1 Answer Rat-trap bond: Non-load bearing wall \u2013 cheaper to construct than other walls but does not support roof, floor or other structures. Uses roughly half the amount of bricks as Dutch bond. Dutch bond: Load bearing wall \u2013 supports roof, floors or other structures. Is stronger than rat-trap but uses twice the amount of bricks.\n\nManaging a virtual team - how would you approach this when you have troubled team member? 1 Answer Basically to take prompt action in that the situation not be",
      "index": 0,
      "finish_reason": "length",
      "logprobs": null
    }
  ],
  "usage": {
    "completion_tokens": 100,
    "prompt_tokens": 22,
    "total_tokens": 122
  }
}
'''


# Get query from command line arguments
query = sys.argv[1:]


# Start building vector db
input_size = 4096
outputs = 512
chunk_overlap = 20
chunk_size = 1000
file_path = os.getcwd() + '/docs/ml_project1_data.csv'
docs = CSVLoader(file_path).load()
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
documents = text_splitter.split_documents(docs)
# index = Chroma.from_documents(documents, embedding_function)



# Get response from Azure AI
response = openai.Embedding.create(input = query, engine = deployment_name)

response_pretty = response['data'][0]['embedding']
print(response_pretty)




