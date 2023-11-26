from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import json

def generate_text():
    model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    # Prepare the text
    text = "This is an example sentence."
    inputs = tokenizer(text, return_tensors='np')
    output = model.encode(text, convert_to_tensor=True)

    # Convert the tensor to a list
    output_list = output.tolist()

    # Save the list to a JSON file
    with open('tests/data/text_embeddings.json', 'w') as f:
        json.dump(output_list, f)

def generate_image():
    model = 