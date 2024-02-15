import requests

# Define the endpoint UR

def upload():
    url = 'http://localhost:5000/upload_docs'
    # Path to the files you want to upload
    file_paths = ["C:\\Users\\Ashish\\Downloads\\Report 1.pdf"]

    # Prepare the files to be sent
    files = []
    for file_path in file_paths:
        files.append(('documents', (file_path, open(file_path, 'rb'))))

    try:
        # Send the request to the server
        response = requests.post(url, files=files)

        # Check the response
        if response.status_code == 200:
            print('Files uploaded successfully!')
        else:
            print('Error:', response.json())
    except Exception as e:
        print('An error occurred:', str(e))


# Function to send a query to the server and get the response
def send_query(query):
    url = 'http://localhost:5000/chat'
    try:
        # Prepare the request data
        data = {'query': query}

        # Send the request to the server
        response = requests.post(url, json=data)

        # Check the response
        if response.status_code == 200:
            result = response.json()
            return result['response']
        else:
            return 'Error: ' + str(response.status_code)
    except Exception as e:
        return 'An error occurred: ' + str(e)

# Main function for user interaction
def ask():
    print("Welcome to the chat interface!")
    print("Type 'exit' to end the chat.")
    while True:
        # Take user input
        query = input("You: ")

        # Exit condition
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        # Send the query to the server and get the response
        response = send_query(query)

        # Print the response
        print("Bot:", response)

# Run the main function
if __name__ == "__main__":
    upload()
    ask()
    
