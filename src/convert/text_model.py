import subprocess

def convert():
    subprocess.run(["python", 
                    "-m", 
                    "pip",
                    "install",
                    "git+https://github.com/huggingface/optimum.git"])
    subprocess.run(["optimum-cli", 
                    "export","onnx", 
                    "-m", "sentence-transformers/clip-ViT-B-32-multilingual-v1", 
                    "--task", 
                    "feature-extraction", 
                    "models/text"])   