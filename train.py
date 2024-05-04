# Import necessary modules and functions
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm.rich import tqdm_rich
from google.cloud import storage

# Assuming Encoder, Decoder, Seq2Seq, and load_data are defined in models.py and data.py respectively
from models import Encoder, Decoder, Seq2Seq
from data import load_data

# Constants
N_EPOCHS = float('inf')  # No upper limit for epochs
LEARNING_RATE = 1e-6  # As low as possible
BATCH_SIZE = 100  # Fixed batch size

# Evaluate the model
def evaluate(model, test_loader, criterion):
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            output = model(x)
            
            loss = criterion(output, y)
            epoch_loss += loss.item()
            
    return epoch_loss / len(test_loader)

# Translate a sequence
def translate_sequence(sequence, model):
    output = model(sequence)
    
    return output

# ... use the translate_sequence function ...


from tqdm import tqdm
from google.cloud import storage

from tqdm.rich import tqdm_rich

def train_model(resume=False):
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')
    # Load and preprocess the data ...
    train_loader, test_loader, input_dim, output_dim, PAD_IDX, source_code_files, asm_files = load_data(bucket_name='chrig')
    
    # Make sure the data loader is distributed
    train_sampler = DistributedSampler(train_loader)
    test_sampler = DistributedSampler(test_loader)
    # Define the dimensions of your input and output data, as well as the dimensions of your model's hidden layers
    emb_dim = 256
    hid_dim = 512
    n_layers = 2
    dropout = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model
    encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    # Make sure the model is distributed
    model = DistributedDataParallel(model)
    # Define the optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    start_epoch = 0
    if resume:
        checkpoint = torch.load('checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    # Train the model
    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        
        epoch_loss = 0
        
        with tqdm_rich(total=len(train_loader), desc=f'Epoch {epoch+1}', unit='batch') as pbar:
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                output = model(x)
                
                loss = criterion(output, y)
                loss.backward()
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                pbar.set_postfix({'loss': epoch_loss / (i+1)})
                pbar.update()
        
        # Save the model and optimizer state, along with the current epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pt')
    # Delete the source code and its associated asm files from the Google Cloud bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('chrig')
    for file_name in source_code_files + asm_files:
        blob = bucket.blob(file_name)
        blob.delete()