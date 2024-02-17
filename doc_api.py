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

##### for creating a simple story-book
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import black, white
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate
import google.generativeai as genai
import re

##### for stable-diffusion
import time
import requests
import random
from PyPDF2 import PdfMerger, PdfReader
from datetime import datetime
import base64
warnings.filterwarnings("ignore")

# bat_directory = "D:\\new_ComfyUI_windows_portable_nvidia_cu121_or_cpu\\ComfyUI_windows_portable"
# bat_file_path = "D:\\new_ComfyUI_windows_portable_nvidia_cu121_or_cpu\\ComfyUI_windows_portable\\run_nvidia_gpu.bat"
# def run_bat_file():
#     # Change the working directory to the .bat file location
#     os.chdir(bat_directory)
#     # Execute the .bat file
#     os.system(bat_file_path)
# thread = threading.Thread(target=run_bat_file)
# # thread.start()

# # try:
# #     time.sleep(60)
# # except Exception as e:
# #     print(e)

## chatwithdoc GEMAI config STARTS ------------------------------
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
## chatwithdoc GENAI config ENDS ------------------------------------

## NOVEL-GENERATION config STARTS -----------------------------------
NOVEL_OUTPUT_DIR = "E:\\chat_with_doc\\novels"
NOVEL_IMAGES_DIR = "E:\\discord_bot\\output"
NOVEL_IMAGE_PATHS = []
MERGE_PDF_PATHS = []
## NOVEL-GENERATION config ENDS -------------------------------------

## IMAGE-GENERATION config STARTS -----------------------------------
IMAGE_API = "https://d8f5-103-94-57-212.ngrok-free.app/process_prompt"
# workflow_directory = "E:\\discord_bot\\workflows\\"
# workflow_file = "workflow_api.json"
# workflow_path = workflow_directory+workflow_file
# URL = "http://127.0.0.1:8188/prompt"
IMAGINE_OUTPUT_DIR = "E:\\discord_bot\\output"

with open("E:\discord_bot\credentials\credentials.json", 'r') as file:
    data = json.load(file)
TOKEN = data["token"]
## IMAGE-GENERATION config ENDS -------------------------------------


### methods for dealing with novel creation STARTS HERE <<-------------------------------------
def add_image_to_pdf(pdf_path, image_paths, contents):
    # Create an empty PDF with landscape orientation
    global MERGE_PDF_PATHS
    print("CONTENT--->",contents)
    print("Paths--->",image_paths)
    # print("LENGTH:",len(image_paths),len(contents))
    path = pdf_path
    for i in range(len(image_paths)):
        print('hello')
        random_integer = random.randint(1, 10000)
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        current_datetime+=str(random_integer)
        # current_datetime = datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3]

        pdf_path =path+''+(current_datetime)+'.pdf'
        MERGE_PDF_PATHS.append(pdf_path)
        c = canvas.Canvas(pdf_path, pagesize=landscape(letter))
        image_path = image_paths[i]
        content = contents[i]

        # Add image to the background of the first page with increased transparency
        img = ImageReader(image_path)
        c.drawImage(img, 0, 0, width=11*inch, height=8.5*inch, mask='auto')

        # Split content into words
        words = content.split()
        max_words_per_line = 15

        # Check if content exceeds 20 words per line
        if len(words) > max_words_per_line:
            # Attempt to fit content with reduced font size
            max_font_size = 40  # Adjust as needed
            min_font_size = 8   # Adjust as needed
            font_name = "Helvetica-Bold"

            # Reduce font size until content fits within page width
            for font_size in range(max_font_size, min_font_size, -1):
                c.setFont(font_name, font_size)
                lines = []
                current_line = ''
                for word in words:
                    if len(current_line.split()) < max_words_per_line:
                        current_line += ' ' + word
                    else:
                        lines.append(current_line.strip())
                        current_line = word
                # for word in words:
                #     if len(current_line.split()) < max_words_per_line and c.stringWidth(current_line + ' ' + word) < 11 * inch:
                #         current_line += ' ' + word
                #     else:
                #         lines.append(current_line.strip())
                #         current_line = word
               

                lines.append(current_line.strip())
                total_width = max([c.stringWidth(line) for line in lines])
                if total_width < 11 * inch:
                    break

        else:
            # Font size determination for content with less than 20 words
            font_size = 15  # Adjust as needed
            font_name = "Helvetica-Bold"
            c.setFont(font_name, font_size)

        # Calculate y_offset for center alignment
        total_text_height = len(lines) * font_size if len(words) > max_words_per_line else font_size
        y_offset = (8.5*inch - total_text_height) / 2

        # Write content in the foreground with improved visibility and center alignment
        c.setFillColorRGB(250, 250, 250)  # White color for text
        for line in lines:
            print("Line:",line)
            # Draw the text with black outline
            c.setFillColorRGB(0, 0, 0)  # Black color for outline
            c.drawString((11*inch - c.stringWidth(line)) / 2 - 1, y_offset, line)  # Offset to apply the outline
            c.drawString((11*inch - c.stringWidth(line)) / 2 + 1, y_offset, line)  # Offset to apply the outline
            c.drawString((11*inch - c.stringWidth(line)) / 2, y_offset - 1, line)  # Offset to apply the outline
            c.drawString((11*inch - c.stringWidth(line)) / 2, y_offset + 1, line)  # Offset to apply the outline
            # Draw the text with white fill
            c.setFillColorRGB(1, 1, 1)  # White color for text
            c.drawString((11*inch - c.stringWidth(line)) / 2, y_offset, line)
            y_offset -= font_size
        c.save()
        # Save the canvas
def get_raw_story(idea):

    genai.configure(api_key="AIzaSyD0aUP2_GzokzvP6FYswoD2rsjCZtJGt98")

    # Set up the model
    generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 2048,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)

    instruction = f"""Generate the story based on the given idea. It is going to be a small visual-story book\n
                    Split your story into short paragraphs, suitable for a visual storybook and enclose it within '$$' symbols\n
                    Generate a detailed image description of the scenery (in Stable Diffusion format) to match each paragraph\n
                     remember, Enclose the image description between triple underscores (i.e. ___example prompt___)\n
                     Do not provide any additional description like part number of page number for story parts, just write it enclosed in '$$' symbol\n
                    The story has to be meaningful and continuous in meaning, so don't make it too small either.\n
                    Here is the idea:{idea}. Ensure that the text description is enclosed within '$$' and image prompt is enclosed within ___ (i.e. ___prompt___)\n
                """
    

    prompt_parts = [instruction
    ]

    
    response = model.generate_content(prompt_parts)
    return response.text  
def extract_descriptions(text):
    text_descriptions = re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL)
    prompt_descriptions = re.findall(r'___(.*?)___', text)
    print("")
    return text_descriptions, prompt_descriptions
def merge_pdfs(paths, output_filename):
    """Merges PDF files from a list of paths into a single PDF.

    Args:
        paths (list): A list of file paths to PDF files.
        output_filename (str): The name of the merged PDF file.
    """

    merger = PdfMerger()
    print("merger_PATHS:",paths)
    for path in paths:
        if not path.lower().endswith('.pdf'):
            print(f"Skipping non-PDF file: {path}")
            continue

        try:
            with open(path, 'rb') as infile:
                merger.append(infile)
        except FileNotFoundError:
            print(f"Error: File not found: {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}")

    with open(output_filename, 'wb') as outfile:
        merger.write(outfile)

    print(f"PDF files merged successfully into '{output_filename}'")

### methods for dealing with novel creation ENDS HERE <<-------------------------------------

### methods for dealing with chat_with_doc STARTS HERE <----------------------------------------
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
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
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
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
### methods for dealing with chat_with_doc ENDS HERE <---------------------------------------

### methods for dealing with IMAGE-GENERATION STARTS HERE<--------------------------
def get_latest_image(folder):
    files = os.listdir(folder)
    image_files = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg'))]
    image_files.sort(key=lambda x:os.path.getmtime(os.path.join(folder,x)))
    latest_image =  os.path.join(folder, image_files[-1]) if image_files else None
    # print(latest_image)
    return latest_image
def start_queue(prompt_workflow):
    p = {"prompt":prompt_workflow}
    data = json.dumps(p).encode("utf-8")
    requests.post(URL,data=data)
def generate_images(image_prompts):
    for image_prompt in image_prompts:
      workflow = ''
      with open(workflow_path, "r") as file_json:
          workflow = json.load(file_json)
          workflow["6"]["inputs"]["text"] = image_prompt
          workflow["3"]["inputs"]["seed"] += 1
          with open(workflow_path,"w") as json_file:
              json.dump(workflow,json_file, indent=2)
      
      previous_image = get_latest_image(IMAGINE_OUTPUT_DIR)
      start_queue(workflow)
      while get_latest_image(IMAGINE_OUTPUT_DIR)==previous_image:
          try:
              time.sleep(3)
          except Exception as e:
              print(e)
          continue
      image_path = get_latest_image(IMAGINE_OUTPUT_DIR)
      NOVEL_IMAGE_PATHS.append( image_path)
def save_png(png_content, output_folder="E:\\discord_bot\\output"):
    # Create the directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # print("PNGCONTENT:",png_content)
    # Generate a filename based on the current datetime
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = os.path.join(output_folder, f"{current_datetime}.png")
    # print(output_filename)
    # Write the PDF content to the file
    with open(output_filename, "wb") as png_file:
        png_file.write(png_content)
    # return output_filename

def api_imagine(image_prompts):
    global NOVEL_IMAGE_PATHS
    # print("IMG_PROMPTS:",image_prompts)
    for prompt in image_prompts:
        prompt_dict = {'prompt':prompt}
        response = requests.post(IMAGE_API,json=prompt_dict)
        response = response.json()
        image_content = base64.b64decode(response['imageB64'])
        # png_content = base64.b64decode(image_content)
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = os.path.join(IMAGINE_OUTPUT_DIR, f"{current_datetime}.png")
        NOVEL_IMAGE_PATHS.append(output_filename)
        with open(output_filename, 'wb') as f:
            f.write(image_content)
                # save_png(png_content)
### methods for dealing with IMAGE-GENERATION ENDS HERE<--------------------------

## chatwithdoc ROUTES START----------------------------------------------------
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
        # print("PATHS:",PATHS)
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
# chatwithdoc ROUTES END ------------------------------------------------

# image-generation thread--------------------
    
# NOVEL-GENERATION ROUTES START ------------------------------------------
@app.route('/generate_novel',methods=['POST'])
def generate_novel():
    data = request.json
    idea = data['idea']
    print("IDEA:",idea)
    raw_story = get_raw_story(idea)
    print("RAW_STORY:",raw_story)
    story_parts, image_prompts = extract_descriptions(raw_story)
    global NOVEL_IMAGE_PATHS
    global MERGE_PDF_PATHS
    NOVEL_IMAGE_PATHS = []
    api_imagine(image_prompts=image_prompts)
    add_image_to_pdf("E:\\chat_with_doc\\novels\\testing",NOVEL_IMAGE_PATHS,story_parts)
    merge_pdfs(MERGE_PDF_PATHS,"E:\\chat_with_doc\\merged_novel\\story.pdf")
    print("MERGEPDFPATHS:",MERGE_PDF_PATHS,"NOVELIMAGEPATHS:",NOVEL_IMAGE_PATHS)
    MERGE_PDF_PATHS = []
    return "Hello"
if __name__ == '__main__':
    app.run(debug=True)