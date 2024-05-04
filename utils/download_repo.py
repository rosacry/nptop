import os
import time
import logging
import requests
import zipfile
import shutil
from google.cloud import storage
from dotenv import load_dotenv
from tqdm.rich import tqdm_rich

load_dotenv()  # load environment variables from .env file

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "adept-stage-422221-u0-23df2eb5efb0.json")

def download_repo(user, repo, token, storage_client, bucket_name):
    try:
        # Fetch repository information
        repo_info_url = f"https://api.github.com/repos/{user}/{repo}"
        headers = {'Authorization': f'token {token}'}
        repo_info_r = requests.get(repo_info_url, headers=headers)
        repo_info = repo_info_r.json()
        # Get the default branch of the repository
        default_branch = repo_info.get('default_branch', 'master')
        # Download the repository
        url = f"https://api.github.com/repos/{user}/{repo}/zipball/{default_branch}"
        r = requests.get(url, headers=headers)
        
        if r.status_code == 404:
            logging.error(f"Repository {user}/{repo} does not exist or the {default_branch} branch is not available.")
            return
        if r.status_code == 403:
            logging.error(f"Rate limit exceeded when trying to download {user}/{repo}. Pausing for 1 hour.")
            time.sleep(3600)  # Pause for 1 hour
            return
        with open('repo.zip', 'wb') as f:
            f.write(r.content)
        
        with zipfile.ZipFile('repo.zip', 'r') as zip_ref:
            zip_ref.extractall('data/source_code_files')
        
        # Upload the repository to Google Cloud Storage
        bucket = storage_client.get_bucket(bucket_name)
        for root, dirs, files in os.walk('data/source_code_files'):
            for file in files:
                local_file = os.path.join(root, file)
                remote_file = local_file.replace('data/source_code_files', f'{bucket_name}/source_code_files/{user}/{repo}')
                blob = bucket.blob(remote_file)
                blob.upload_from_filename(local_file)
        
        logging.info(f"Successfully downloaded and uploaded {user}/{repo}.")
    except Exception as e:
        logging.error(f"An error occurred when processing {user}/{repo}: {e}")
    finally:
        if os.path.exists('repo.zip'):
            os.remove('repo.zip')
        if os.path.exists('data/source_code_files'):
            shutil.rmtree('data/source_code_files')


def get_popular_repos(token, num_repos=100):
    url = f"https://api.github.com/search/repositories?q=stars:%3E1&sort=stars&order=desc"
    headers = {'Authorization': f'token {token}'}
    r = requests.get(url, headers=headers)
    items = r.json()['items']
    return [(item['owner']['login'], item['name']) for item in items[:num_repos]]

token = os.getenv('GITHUB_TOKEN')  # Get the GitHub token from environment variable

# Initialize a storage client
storage_client = storage.Client()

# Specify the name of your bucket
bucket_name = 'chrig'

# Get the most popular repositories
repos = get_popular_repos(token)

# Download each repository
for user, repo in tqdm_rich(repos, desc='Downloading Repositories', unit='repo'):
    download_repo(user, repo, token, storage_client, bucket_name)