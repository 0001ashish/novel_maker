from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
import google.generativeai as genai
import re

def add_image_to_pdf(pdf_path, image_path, content):
    # Create an empty PDF with landscape orientation
    c = canvas.Canvas(pdf_path, pagesize=landscape(letter))

    # Add image to the background of the first page with increased transparency
    img = ImageReader(image_path)
    c.drawImage(image_path, 0, 0, width=11*inch, height=8.5*inch, mask='auto')

    # Split content into words
    words = content.split()
    max_words_per_line = 20

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
            lines.append(current_line.strip())
            total_width = max([c.stringWidth(line) for line in lines])
            if total_width < 11 * inch:
                break

    else:
        # Font size determination for content with less than 20 words
        font_size = 40  # Adjust as needed
        font_name = "Helvetica-Bold"
        c.setFont(font_name, font_size)

    # Calculate y_offset for center alignment
    total_text_height = len(lines) * font_size if len(words) > max_words_per_line else font_size
    y_offset = (8.5*inch - total_text_height) / 2

    # Write content in the foreground with improved visibility and center alignment
    c.setFillColorRGB(250, 250, 250)  # White color for text
    for line in lines:
        # Draw the text with black outline
        c.setFillColorRGB(0, 0, 0)  # Black color for outline
        c.drawString((11*inch - c.stringWidth(line)) / 2 - 1, y_offset, line)  # Offset to apply the outline
        c.drawString((11*inch - c.stringWidth(line)) / 2 + 1, y_offset, line)  # Offset to apply the outline
        c.drawString((11*inch - c.stringWidth(line)) / 2, y_offset - 1, line)  # Offset to apply the outline
        c.drawString((11*inch - c.stringWidth(line)) / 2, y_offset + 1, line)  # Offset to apply the outline
        # Draw the text with white fill
        c.setFillColorRGB(1, 1, 1)  # White color for text
        c.drawString((11*inch - c.stringWidth(line)) / 2, y_offset, line)
        y_offset += font_size

    # Save the canvas
    c.save()

def get_raw_story(idea):

    genai.configure(api_key="GEMINI_API_KEY") #<--- replace with your Gemini API key

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
                    Generate a detailed image description of the scenery excluding charactar description(in Stable Diffusion format) to match each paragraph\n
                     remember, Enclose the image description between triple underscores (i.e. ___example prompt___)\n
                     Do not provide any additional description like part number of page number for story parts, just write it enclosed in '$$' symbol\n
                    The story has to be meaningful and continuous in meaning, so don't make it too small either.\n
                    Avoid using gender-specific descriptors such as 'girls,' 'boys,' 'women,' or 'men' in descriptions.\n
                    Here is the idea:{idea}. Be carefull with '$$' and ___prompt___ formats
                """
    

    prompt_parts = [instruction
    ]

    
    response = model.generate_content(prompt_parts)
    return response.text

def extract_descriptions(text):
    text_descriptions = re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL)
    prompt_descriptions = re.findall(r'___(.*?)___', text)
    return text_descriptions, prompt_descriptions


if __name__ == "__main__":
    pdf_path = "output4.pdf"
    image_path = "C:\\Users\\Ashish\\Pictures\\Saved Pictures\\00148-2651771783.png"
    content = "What a beautiful treehouse ! I would love to live there. Would u like to come with me there ? What a beautiful treehouse ! I would love to live there. Would u like to come with me there ? What a beautiful treehouse ! I would love to live there. Would u like to come with me there ?What a beautiful treehouse ! I would love to live there. Would u like to come with me there ? What a beautiful treehouse ! I would love to live there. Would u like to come with me there ? What a beautiful treehouse ! I would love to live there. Would u like to come with me there ?"
    story = get_raw_story("Vacation at a beautiful treehouse")
    parts, prompts = extract_descriptions(story)
    # add_image_to_pdf(pdf_path, image_path, content)
    # print(f"PDF with image and content created: {pdf_path}")
    print(story)
    print(parts,'\n',prompts)
