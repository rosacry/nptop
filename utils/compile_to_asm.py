import dis
import os
import subprocess
from google.cloud import storage
from tqdm.rich import tqdm_rich

def compile_to_asm(lang, source_file_path, asm_file_path):
    # Initialize a storage client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.get_bucket('chrig')
    # Get the source code blob
    source_blob = bucket.blob(source_file_path)
    # Download the source code as text
    source_code = source_blob.download_as_text()
    # Save the source code to a temporary file
    with open('temp_source_file', 'w') as file:
        file.write(source_code)
    # Compile the source code to assembly
    langs = ['c', 'cpp', 'java', 'rust', 'csharp', 'python']
    for lang in tqdm_rich(langs, desc='Compiling to Assembly', unit='lang'):
        if lang == 'c':
            subprocess.run(["gcc", "-S", "-o", 'temp_asm_file', 'temp_source_file'], check=True)
        elif lang == 'cpp':
            subprocess.run(["g++", "-S", "-o", 'temp_asm_file', 'temp_source_file'], check=True)
        elif lang == 'java':
            class_file_path = 'temp_source_file'.replace('.java', '.class')
            subprocess.run(["javac", "-d", ".", 'temp_source_file'], check=True)
            subprocess.run(["jad", "-o", "-d", ".", class_file_path], check=True)
            os.rename('temp_source_file'.replace('.java', '.jad'), 'temp_asm_file')
        elif lang == 'rust':
            subprocess.run(["rustc", "--emit", "asm", "-o", 'temp_asm_file', 'temp_source_file'], check=True)
        elif lang == 'csharp':
            dll_file_path = 'temp_source_file'.replace('.cs', '.dll')
            subprocess.run(["csc", "-out:" + dll_file_path, 'temp_source_file'], check=True)
            subprocess.run(["ildasm", "/OUT=" + 'temp_asm_file', dll_file_path], check=True)
        elif lang == 'python':
            with open('temp_source_file', 'r') as file:
                code = file.read()
            bytecode = dis.Bytecode(code)
            with open('temp_asm_file', 'w') as file:
                file.write(str(bytecode.dis()))
        else:
            raise ValueError(f"Unsupported language: {lang}")
    # Read the assembly code from the temporary file
    with open('temp_asm_file', 'r') as file:
        asm_code = file.read()
    # Save the assembly code to a blob
    asm_blob = bucket.blob(asm_file_path)
    asm_blob.upload_from_string(asm_code)