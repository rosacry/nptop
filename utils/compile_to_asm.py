import dis
import os
import subprocess

#Compiling Files to assembly code to train the ai to understand precompiled code

def compile_to_asm(lang, source_file_path, asm_file_path):
    if lang == 'c':
        subprocess.run(["gcc", "-S", "-o", asm_file_path, source_file_path], check=True)
    elif lang == 'cpp':
        subprocess.run(["g++", "-S", "-o", asm_file_path, source_file_path], check=True)
    elif lang == 'java':
        class_file_path = source_file_path.replace('.java', '.class')
        subprocess.run(["javac", "-d", ".", source_file_path], check=True)
        subprocess.run(["jad", "-o", "-d", ".", class_file_path], check=True)
        os.rename(source_file_path.replace('.java', '.jad'), asm_file_path)
    elif lang == 'rust':
        subprocess.run(["rustc", "--emit", "asm", "-o", asm_file_path, source_file_path], check=True)
    elif lang == 'csharp':
        dll_file_path = source_file_path.replace('.cs', '.dll')
        subprocess.run(["csc", "-out:" + dll_file_path, source_file_path], check=True)
        subprocess.run(["ildasm", "/OUT=" + asm_file_path, dll_file_path], check=True)
    elif lang == 'python':
        with open(source_file_path, 'r') as file:
            code = file.read()
        bytecode = dis.Bytecode(code)
        with open(asm_file_path, 'w') as file:
            file.write(str(bytecode.dis()))
    else:
        raise ValueError(f"Unsupported language: {lang}")
