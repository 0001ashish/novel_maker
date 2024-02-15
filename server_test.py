from flask import Flask, request
import os

app = Flask(__name__)

@app.route('/upload_docs', methods=['POST'])
def upload_docs():
    try:
        uploaded_files = request.files.getlist('documents')
        print(uploaded_files)
        # Specify the path where you want to store the uploaded documents
        storage_path = 'E:\\chat_with_doc\\documents'

        # Create the storage directory if it doesn't exist
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        for file in uploaded_files:
            file_path = os.path.join(storage_path, file.filename)
            print(file_path)
            file.save(file_path)

        return "Documents uploaded successfully."

    except Exception as e:
        print(f"Error: {e}")
        return "Error uploading documents.", 500

if __name__ == '__main__':
    app.run(debug=True)
