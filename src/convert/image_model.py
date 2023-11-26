import subprocess

def convert():
    subprocess.run(["optimum-cli", 
                    "export","onnx", 
                    "-m", "patrickjohncyh/fashion-clip", 
                    "--trust-remote-code", 
                    "--framework" ,"pt", 
                    "models/image"])    