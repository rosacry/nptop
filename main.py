from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from train import translate_sequence, train_model
from utils.download_repo import download_repo
from tqdm.rich import tqdm_rich
from rich.console import Console
from rich.progress import Progress
import torch
import os

console = Console()
progress = Progress(console=console)

def build_vocab(code):
    with progress.start_task(total=len(code.split())) as task:
        console.print("Building vocabulary...", style="bold magenta")
        # Tokenize the code
        tokens = code.split()

        # Create a set of unique tokens
        vocab = set()
        for token in tokens:
            vocab.add(token)
            progress.update(task, advance=1)

    return vocab

def load_model():
    console.print("Loading model...", style="bold magenta")
    # Load your assembly and high-level code
    assembly_code = open('path/to/assembly_code.asm').read()
    high_level_code = open('path/to/high_level_code.py').read()

    # Build your vocabularies
    assembly_vocab = build_vocab(assembly_code)
    high_level_vocab = build_vocab(high_level_code)
    # Define the dimensions of your input and output data, as well as the dimensions of your model's hidden layers
    input_dim = len(assembly_vocab)  # Replace with the size of your assembly language vocabulary
    output_dim = len(high_level_vocab)  # Replace with the size of your high-level language vocabulary
    emb_dim = 256
    hid_dim = 512
    n_layers = 2
    dropout = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate your models
    encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    # Check if a pre-trained model exists
    if os.path.exists('model.pt'):
        # Load the model
        model.load_state_dict(torch.load('model.pt'))
    else:
        # Train the model
        n_epochs = 10  # Define the number of epochs
        train_model()
        model.load_state_dict(torch.load('model.pt'))

    return model

def main():
    console.print("Starting main function...", style="bold magenta")
    # Download open-source code
    repo_url = 'https://github.com/user/repo.git'  # Replace with the actual repo URL
    download_repo(repo_url)

    model = load_model()

    # Decompile an assembly code file
    asm_file = 'path/to/asm_file.asm'  # Replace with the actual file path
    with open(asm_file, 'r') as f:
        sequence = f.read()
    sequence = torch.tensor([int(x) for x in sequence.split()])  # Convert the sequence to a tensor
    sequence = sequence.to(model.device)
    decompiled_code = translate_sequence(sequence, model)

    console.print("Decompiled code:", style="bold green")
    console.print(decompiled_code)

if __name__ == "__main__":
    main()