from sentence_transformers import SentenceTransformer
from fashion_clip.fashion_clip import FashionCLIP
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

def generate_image_():
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    image = Image.open("tests/data/test_image.jpg")

    inputs = processor(text=["this is a hat"],images=image, return_tensors="pt", padding=True)
    print(inputs.keys())

    outputs = model(**inputs)
    embeddings = outputs.image_embeds  # this is the image-text similarity score
    print(embeddings)
    with open('tests/data/image_embeddings.json', 'w') as f:
        json.dump(embeddings.tolist()[0], f)

def generate_image():
    fclip = FashionCLIP('fashion-clip')
    image = Image.open("tests/data/test_image.jpg")
    # we create image embeddings and text embeddings
    image_embeddings = fclip.encode_images([image], batch_size=32)
    print(image_embeddings)
    with open('tests/data/image_embeddings.json', 'w') as f:
        json.dump(image_embeddings.tolist()[0], f)