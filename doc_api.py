from IPython.display import display
from IPython.display import Markdown
import textwrap
import os
import json
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import shutil

warnings.filterwarnings("ignore")

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

with open("E:\\Serfi\\credentials\\credentials.json","r") as file:
  data = json.load(file)

GOOGLE_API_KEY= data["genai_api_key"]
genai.configure(api_key=GOOGLE_API_KEY)

model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.3,convert_system_message_to_human=True)

app = Flask(__name__)

UPLOAD_FOLDER = 'E:\\chat_with_doc\\documents'
ALLOWED_EXTENSIONS = {'pdf'}
PATHS = []
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
qa_chain=None
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_dir():
    global PATHS
    for doc in os.listdir(UPLOAD_FOLDER):
      file_path = os.path.join(UPLOAD_FOLDER,doc)
  
      try:
          if os.path.isfile(file_path):
              os.unlink(file_path)
              print(f"Deleted: {file_path}")
      except Exception as e:
          print(f"Error deleting {file_path}: {e}")
    PATHS= []
    
def get_text(paths):
  pages=[]
  for path in paths:
    pdf_loader = PyPDFLoader(path)
    pages +=pdf_loader.load_and_split()
    print("PAGELEN:",len(pages))

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
  context = "\n\n".join(str(p.page_content) for p in pages)
  texts = text_splitter.split_text(context)
  return texts

@app.route('/upload_docs', methods=['POST'])
def upload_docs():
    clear_dir()
    try:
        if 'documents' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        files = request.files.getlist('documents')
        # print("111\n")
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # print("FILENAME:",filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # print("PATH OF FILE IS:-->",file_path)
                PATHS.append(file_path)
                file.save(file_path)
        print("PATHS:",PATHS)
        # print("222\n")
        texts = get_text(PATHS)
        # print(texts)
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
        # print("VECTOR INDEX:",vector_index)
        global qa_chain
        qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=vector_index,
            return_source_documents=True

        )
        return jsonify({'success': True, 'result': "Your document has been uploaded successfully"}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat',methods=['POST'])
def chat():
  data = request.json
  query = data.get('query', '')
  result = qa_chain({"query": query})
  print("result:",result["result"])
  return {'response':(result["result"])}

if __name__ == '__main__':
    app.run(debug=True)