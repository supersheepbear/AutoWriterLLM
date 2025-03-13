"""API client module for AutoWriterLLM.

This module handles the interaction with various LLM API providers.
"""

import logging
import time
import requests
import httpx
import anthropic
import threading
from openai import OpenAI
from typing import Any, Dict, Optional, List

from autowriterllm.exceptions import ContentGenerationError
from autowriterllm.models import RateLimitConfig

logger = logging.getLogger(__name__)


class APIRateLimiter:
    """Handles API rate limiting and retries.
    
    This class manages API rate limiting to avoid exceeding provider limits.
    It keeps track of request times and enforces waiting periods when necessary.
    
    Attributes:
        config (RateLimitConfig): Configuration for rate limiting
        request_times (List[float]): List of timestamps for recent requests
    """
    
    def __init__(self, config: RateLimitConfig):
        """Initialize the rate limiter.
        
        Args:
            config: Configuration for rate limiting
        """
        self.config = config
        self.request_times: List[float] = []
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Wait if we're exceeding rate limits."""
        with self._lock:
            now = time.time()
            # Remove old requests
            self.request_times = [t for t in self.request_times 
                                if now - t < 60]
            
            if len(self.request_times) >= self.config.max_requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            
            self.request_times.append(now)
            logger.debug(f"Current request count: {len(self.request_times)}/{self.config.max_requests_per_minute}")


class APIClientFactory:
    """Factory for creating API clients.
    
    This class creates the appropriate API client based on the provider configuration.
    
    Attributes:
        config (Dict): Configuration dictionary
    """
    
    def __init__(self, config: Dict):
        """Initialize the factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def create_client(self, provider_name: Optional[str] = None) -> Any:
        """Create an API client for the specified provider.
        
        Args:
            provider_name: Name of the provider (uses default if None)
            
        Returns:
            Any: The initialized API client
            
        Raises:
            ValueError: If the provider configuration is invalid
        """
        if provider_name is None:
            provider_name = self.config.get("provider")
            if not provider_name:
                raise ValueError("No default provider specified in configuration")
        
        try:
            # Get provider configuration
            providers_config = self.config.get("providers", {})
            if provider_name not in providers_config:
                raise ValueError(f"Provider '{provider_name}' not found in configuration")
                
            provider_config = providers_config[provider_name]
            api_key = provider_config.get("api_key")
            base_url = provider_config.get("base_url")
            
            if not api_key:
                raise ValueError(f"{provider_name.title()} API key not found in config")
                
            # Configure custom httpx client with longer timeouts
            http_client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=10.0,    # connection timeout
                    read=300.0,      # read timeout
                    write=20.0,      # write timeout
                    pool=10.0        # pool timeout
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0
                )
            )
            
            # Initialize client based on provider
            if provider_name == "anthropic":
                logger.info(f"Initializing Anthropic client with base URL: {base_url}")
                return anthropic.Anthropic(
                    api_key=api_key,
                    base_url=base_url if base_url else "https://api.anthropic.com"
                )
            elif provider_name == "openai":
                logger.info(f"Initializing OpenAI client with base URL: {base_url}")
                return OpenAI(
                    api_key=api_key,
                    base_url=base_url if base_url else "https://api.openai.com/v1",
                    http_client=http_client,
                    timeout=300.0
                )
            else:
                # For other providers, use OpenAI client with custom base URL
                logger.info(f"Initializing generic client for {provider_name} with base URL: {base_url}")
                if not base_url:
                    raise ValueError(f"Base URL for provider '{provider_name}' not specified in config")
                    
                return OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    http_client=http_client,
                    timeout=300.0
                )
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            raise


class ContentGenerator:
    """Base class for content generation.
    
    This class provides methods for generating content using various LLM providers.
    
    Attributes:
        client (Any): The API client
        config (Dict): Configuration dictionary
        rate_limiter (APIRateLimiter): Rate limiter for API calls
    """
    
    def __init__(self, config: Dict):
        """Initialize the content generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.client_factory = APIClientFactory(config)
        self.client = self.client_factory.create_client()
        
        # Initialize rate limiter
        rate_limit_config = RateLimitConfig(
            max_requests_per_minute=config.get("rate_limit", {}).get("requests_per_minute", 50)
        )
        self.rate_limiter = APIRateLimiter(rate_limit_config)
    
    def set_model(self, provider_name: str, model_name: str) -> None:
        """Set the current model for content generation.
        
        Args:
            provider_name: The provider name
            model_name: The model name
            
        Raises:
            ValueError: If the provider or model is not found in the configuration
        """
        providers_config = self.config.get("providers", {})
        if provider_name not in providers_config:
            raise ValueError(f"Provider '{provider_name}' not found in configuration")
            
        provider_config = providers_config[provider_name]
        model_found = False
        
        for model in provider_config.get("models", []):
            if model.get("name") == model_name:
                model_found = True
                break
                
        if not model_found:
            raise ValueError(f"Model '{model_name}' not found for provider '{provider_name}'")
            
        # Update current provider and model
        self.config["provider"] = provider_name
        self.config["current_model"] = model_name
        
        # Reinitialize client with new provider
        self.client = self.client_factory.create_client(provider_name)
        logger.info(f"Set model to {provider_name}/{model_name}")
    
    def generate_with_retry(self, prompt: str, max_retries: int = 10) -> str:
        """Generate content with automatic retry on failure.
        
        This method handles the actual API calls to generate content. When generating
        multi-part content, this method is called multiple times with different prompts
        for each part, and the results are combined later.
        
        Args:
            prompt: The prompt to send to the AI
            max_retries: Maximum number of retries on failure
            
        Returns:
            str: The generated content
            
        Raises:
            ContentGenerationError: If content generation fails after all retries
        """
        provider_name = self.config.get("provider")
        providers_config = self.config.get("providers", {})
        provider_config = providers_config.get(provider_name, {})
        
        # Get the current model configuration
        current_model = None
        model_name = self.config.get("current_model")
        
        # Find the current model in the provider's models list
        for model in provider_config.get("models", []):
            if model.get("name") == model_name:
                current_model = model
                break
        
        # If model not found, use the first model in the list
        if not current_model and provider_config.get("models"):
            current_model = provider_config["models"][0]
            model_name = current_model.get("name")
            logger.warning(f"Model {model_name} not found, using {model_name}")
        
        # Get model parameters
        max_tokens = current_model.get("max_tokens") if current_model else self.config.get("max_tokens_per_part", 8192)
        temperature = current_model.get("temperature") if current_model else 0.7
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generation attempt {attempt + 1}/{max_retries}")
                
                if provider_name == "anthropic":
                    logger.debug(f"Using Anthropic API with model {model_name}")
                    response = self.client.messages.create(
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                    logger.info("Successfully generated content from Anthropic API")
                    return content
                    
                elif provider_name == "openai":
                    logger.debug(f"Using OpenAI API with model {model_name}")
                    response = self.client.chat.completions.create(
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.choices[0].message.content
                    logger.info("Successfully generated content from OpenAI API")
                    return content
                    
                else:
                    # Generic API call for other providers
                    logger.debug(f"Using {provider_name} API with model {model_name}")
                    
                    # Try to use the chat completions endpoint (OpenAI-compatible)
                    try:
                        response = self.client.chat.completions.create(
                            model=model_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        content = response.choices[0].message.content
                        logger.info(f"Successfully generated content from {provider_name} API")
                        return content
                    except (AttributeError, NotImplementedError) as e:
                        # If chat completions not available, try direct HTTP request
                        logger.debug(f"Chat completions not available for {provider_name}, using HTTP request: {e}")
                        
                        url = f"{provider_config.get('base_url')}/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {provider_config.get('api_key')}",
                            "Content-Type": "application/json"
                        }
                        
                        data = {
                            "model": model_name,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                        
                        logger.debug(f"Request data: {data}")
                        response = requests.post(url, headers=headers, json=data, timeout=120)
                        
                        if response.status_code != 200:
                            logger.error(f"API error: {response.status_code} - {response.text}")
                            raise ContentGenerationError(f"API error: {response.status_code}")
                        
                        result = response.json()
                        logger.debug(f"API response: {result}")
                        
                        if not result.get("choices"):
                            logger.error("No choices in API response")
                            raise ContentGenerationError("No content in API response")
                        
                        content = result["choices"][0]["message"]["content"]
                        if not content:
                            logger.error("Empty content in API response")
                            raise ContentGenerationError("Empty content in API response")
                        
                        logger.info(f"Successfully generated content from {provider_name} API")
                        return content
                    
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ContentGenerationError(f"Network error after {max_retries} attempts")
                wait_time = min(300, 10 * (2 ** attempt))
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
                
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ContentGenerationError(f"Failed after {max_retries} attempts: {str(e)}")
                wait_time = min(300, 10 * (2 ** attempt))
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        raise ContentGenerationError("Failed to generate content after all retries") 