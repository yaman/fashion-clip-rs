from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,CLIPProcessor, CLIPModel
from PIL import Image
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
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    image = Image.open("tests/data/test_image.jpg")

    inputs = processor(text=["a photo of a red shoe", "a photo of a black shoe"],images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    embeddings = outputs.image_embeds  # this is the image-text similarity score
    print(embeddings.shape)
    with open('tests/data/image_embeddings.json', 'w') as f:
        json.dump(embeddings.tolist()[0], f)
