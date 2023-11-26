from sentence_transformers import models,SentenceTransformer
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import onnx
import numpy as np
import subprocess

class CombinedModel(nn.Module):
    def __init__(self, transformer_model, dense_model):
        super(CombinedModel, self).__init__()
        self.transformer = transformer_model
        self.dense = dense_model

    def forward(self, input_ids, attention_mask):
        transformer_output = self.transformer({'input_ids': input_ids, 'attention_mask': attention_mask})
        token_embeddings = transformer_output['token_embeddings']
        dense_output = self.dense({'sentence_embedding': token_embeddings})
        dense_output_tensor = dense_output['sentence_embedding']
        mean_output = torch.mean(dense_output_tensor, dim=1)
        
        flattened_output = mean_output.squeeze(0)
        return flattened_output

def convert():
    # Load the transformer model
    transformer_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1', cache_folder='models')
    tokenizer = transformer_model.tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/clip-ViT-B-32-multilingual-v1")

    dense_model = models.Dense(
        in_features=768,
        out_features=512,
        bias=False,
        activation_function= nn.Identity()
    )

    # Load the state_dict into the model
    state_dict = torch.load('models/sentence-transformers_clip-ViT-B-32-multilingual-v1/2_Dense/pytorch_model.bin')
    dense_model.load_state_dict(state_dict)
    # Create the combined model
    model = CombinedModel(transformer_model, dense_model)
    model.eval()

    input_text = "This is a multi-lingual version of the OpenAI CLIP-ViT-B32 model. You can map text (in 50+ languages) and images to a common dense vector space such that images and the matching texts are close."

    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Export the model
    torch.onnx.export(model,               # model being run
                    (input_ids, attention_mask), # model input (or a tuple for multiple inputs)
                    "models/model.onnx", # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=17,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input_ids', 'attention_mask'],   # the model's input names
                    output_names = ['embedding'], # the model's output names
                    dynamic_axes={'input_ids': {0 : 'batch_size', 1: 'seq_length'},  
                                    'attention_mask' : {0 : 'batch_size', 1: 'seq_length'},
                                    'output' : {0 : 'batch_size'}})

    onnx.checker.check_model("models/model.onnx")

    comdined_model = onnx.load("models/model.onnx")

    # Get the name and shape of the input
    input_name = comdined_model.graph.input[0].name
    input_shape = [dim.dim_value for dim in comdined_model.graph.input[0].type.tensor_type.shape.dim]
    print(f"Input name: {input_name}, shape: {input_shape}")

    # Get the name and shape of the output
    output_name = comdined_model.graph.output[0].name
    output_shape = [dim.dim_value for dim in comdined_model.graph.output[0].type.tensor_type.shape.dim]
    print(f"Output name: {output_name}, shape: {output_shape}")


    subprocess.run(["rm","-rf","models/text"])

    folder_path = "models/sentence-transformers_clip-ViT-B-32-multilingual-v1"
    subprocess.run(["mv",folder_path,"models/text"])
    subprocess.run(["mv", "models/model.onnx", "models/text/model.onnx"])
    subprocess.run(["rm","-rf","models/text/1_Pooling"])
    subprocess.run(["rm","-rf","models/text/2_Dense"])
    subprocess.run(["rm","-rf","models/text/pytorch_model.bin"])
    # shutil.rmtree(folder_path)