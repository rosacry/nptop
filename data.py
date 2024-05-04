from google.cloud import storage
import os

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "adept-stage-422221-u0-23df2eb5efb0.json"

def load_data(bucket_name='chrig', source_code_dir='source_code_files', asm_dir='asm_files'):
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Get the blobs in the source_code_files and asm_files directories
    source_code_blobs = bucket.list_blobs(prefix=source_code_dir)
    asm_blobs = bucket.list_blobs(prefix=asm_dir)

    # Load the source code and assembly files
    source_code_files = [blob.download_as_text() for blob in source_code_blobs]
    asm_files = [blob.download_as_text() for blob in asm_blobs]

    return source_code_files, asm_files