from gradio_client import Client


client = Client("https://5324024c6b601826b7.gradio.live/")
result = client.predict(
		"Hello!!",	# str  in 'text_input' Textbox component
		"",	# str  in 'pdf_file_content' Textbox component
		api_name="/predict"
)
print(result)