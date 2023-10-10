import sys
import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, LangchainEmbedding
import openai
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader


openai_api_version = '2023-05-15'
OPENAI_API_KEY = 'fd925220485248079613dfc0bbfea02b'

openai.api_key = 'fd925220485248079613dfc0bbfea02b'
openai.api_base = os.getenv('https://ai-recommendations.openai.azure.com/')
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
OPENAI_API_KEY = os.getenv('fd925220485248079613dfc0bbfea02b')

deployment_name = 'genai-task'



llm = AzureOpenAI(deployment_name = deployment_name, openai_api_key = OPENAI_API_KEY, openai_api_version = openai_api_version)

query = 'please list the information associated with id 5524'

# def create_index(file_path):


# prompt_helper = PromptHelper(num_output = outputs, chunk_size_limit = chunk_size)

llm_predictor = LLMPredictor(llm = llm)
# embedding_llm = LangchainEmbedding(OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY))
# embedding_llm = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
# embedding_function = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
# embedding_function = openai.Embedding.create(deployment_name = deployment_name)


# documents = SimpleDirectoryReader(file_path).load_data()
# index = GPTVectorStoreIndex(documents, llm_predictor = llm_predictor, embed_model=embedding_llm, prompt_helper = prompt_helper)
input_size = 4096
outputs = 512
chunk_overlap = 20
chunk_size = 1000
file_path = os.getcwd() + '/docs/ml_project1_data.csv'
docs = TextLoader(file_path).load()
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
documents = text_splitter.split_documents(docs)
response = openai.Embedding.create(input = query, engine = llm)
# index = Chroma.from_documents(documents, embedding_function)


# index.save_to_disk('index.json')
# print('index.json is saved to disk')
print(response)
    

# print(create_index('/docs'))