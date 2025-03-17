from openai import OpenAI
from langchain_openai import AzureChatOpenAI
import logging
import os
import uuid

# Set up logging
log_dir = os.path.join(os.getcwd(), 'tmp', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'custom_llm_debug.log')

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)

class CustomAzureOpenAI(AzureChatOpenAI):
    """Custom Azure OpenAI wrapper with additional headers"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_version = kwargs.get("api_version", "2024-10-21")
        
        # Get base URL from azure_endpoint
        base_url = kwargs.get("azure_endpoint", "").rstrip('/')
        if not base_url:
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
            
        # Add API version to base URL
        self._base_url = f"{base_url}?api-version={api_version}"
        
        logging.debug(f"Initializing CustomAzureOpenAI with base URL: {self._base_url}")
        
        # Create headers dict
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'x-subscription-key': kwargs.get("api_key", ""),
            'x-correlation-id': str(uuid.uuid4())
        }
        
        logging.debug(f"Using headers: {headers}")
        
        # Create client with minimal configuration
        self.client = OpenAI(
            base_url=self._base_url,
            api_key=kwargs.get("api_key", ""),
            default_headers=headers
        ) 