import json
import os
import re
from typing import List, Dict
from abc import ABC, abstractmethod
import requests

# Try importing Hugging Face
try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Try importing Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class LLMProvider(ABC):
    @abstractmethod
    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        pass

    def _create_system_prompt(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> str:
        return f"""You are a language learning assistant. Generate {num_words} {difficulty}-level vocabulary words in {language} related to '{topic}'.
For each word, provide:
1. The word in {language}
2. Its part of speech
3. English translation
4. Definition in {language}
{"5. An example sentence in " + language if include_examples else ""}

Format the response as a JSON array where each item has these fields:
- word: string
- partOfSpeech: string
- translation: string
- definition: string
{"- example: string" if include_examples else ""}"""

class HuggingFaceProvider(LLMProvider):
    def __init__(self):
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Hugging Face package is not installed")
        
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        
        # Initialize client with Mixtral model
        self.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        print(f"Initializing Hugging Face client with model: {self.model}")
        self.client = InferenceClient(token=self.api_key)
        print("Hugging Face client initialized successfully")

    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        system_prompt = self._create_system_prompt(topic, language, num_words, difficulty, include_examples)
        
        # Format prompt for Mixtral
        prompt = f"<s>[INST]{system_prompt}[/INST]"
        
        # Generate response
        print("Generating response with Hugging Face...")
        try:
            response = self.client.text_generation(
                model=self.model,
                prompt=prompt,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )
            print(f"Raw response: {response}")
            
            # Try to find and extract the JSON array
            try:
                # Find the first '[' and last ']' to extract the JSON array
                start = response.find('[')
                end = response.rfind(']') + 1
                if start == -1 or end == 0:
                    raise ValueError("No JSON array found in response")
                
                json_str = response[start:end]
                print(f"Extracted JSON: {json_str}")
                
                vocab_data = json.loads(json_str)
                
                # Ensure it's a list
                if not isinstance(vocab_data, list):
                    raise ValueError("Response is not a list of vocabulary items")
                
                # Validate each item has required fields
                required_fields = ['word', 'translation', 'partOfSpeech', 'definition']
                if include_examples:
                    required_fields.append('example')
                
                for item in vocab_data:
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")
                
                return vocab_data
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                # Try to clean the response and parse again
                cleaned = self._clean_json_string(response)
                print(f"Cleaned JSON: {cleaned}")
                return json.loads(cleaned)
                
        except Exception as e:
            print(f"Error generating vocabulary with Hugging Face: {str(e)}")
            raise ValueError(f"Error generating vocabulary with Hugging Face: {str(e)}")

    def _clean_json_string(self, text: str) -> str:
        """Clean and extract JSON from the response text."""
        # Find the first '[' and last ']'
        start = text.find('[')
        end = text.rfind(']')
        
        if start == -1 or end == -1:
            raise ValueError("Could not find JSON array in response")
        
        # Extract just the array part
        json_text = text[start:end + 1]
        
        # Try to clean up any invalid JSON
        json_text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', json_text)
        json_text = re.sub(r'(?<!\\)\\\'', '\'', json_text)
        json_text = re.sub(r'(?<!\\)\\"', '"', json_text)
        json_text = re.sub(r'\'', '"', json_text)
        
        return json_text

class OllamaProvider(LLMProvider):
    def __init__(self):
        print("Initializing OllamaProvider...")
        if not OLLAMA_AVAILABLE:
            print("Ollama package is not installed")
            raise ImportError("Ollama package is not installed")
        
        # Initialize Ollama client
        self.client = ollama.Client(host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
        print(f"Initialized Ollama client with host: {self.client.host}")
        
        try:
            # Test connection
            print("Testing connection to Ollama...")
            self.client.list()
            print("Successfully connected to Ollama")
        except Exception as e:
            print(f"Could not connect to Ollama server: {str(e)}")
            raise ConnectionError(f"Could not connect to Ollama server: {str(e)}")
        
        try:
            # Pull the model
            print("Pulling Mistral model...")
            self.client.pull('mistral')
            print("Mistral model pulled successfully")
        except Exception as e:
            print(f"Warning: Could not pull model: {str(e)}")
            # Don't raise here, as the model might already be pulled

    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        system_prompt = self._create_system_prompt(topic, language, num_words, difficulty, include_examples)
        
        try:
            print(f"Generating response with Ollama...")
            response = self.client.generate(
                model='mistral',
                prompt=system_prompt
            )
            
            # Get the response text
            response_text = response['response']
            print(f"Raw Ollama response: {response_text}")
            
            # Try to find and extract the JSON array
            try:
                # Find the first '[' and last ']' to extract the JSON array
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start == -1 or end == 0:
                    raise ValueError("No JSON array found in response")
                
                json_str = response_text[start:end]
                print(f"Extracted JSON: {json_str}")
                
                # Parse the JSON
                vocab_data = json.loads(json_str)
                
                # Ensure it's a list
                if not isinstance(vocab_data, list):
                    raise ValueError("Response is not a list of vocabulary items")
                
                # Validate each item has required fields
                required_fields = ['word', 'translation', 'partOfSpeech', 'definition']
                if include_examples:
                    required_fields.append('example')
                
                for item in vocab_data:
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")
                
                return vocab_data
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                # Try to clean the response and parse again
                cleaned = self._clean_json_string(response_text)
                print(f"Cleaned JSON: {cleaned}")
                return json.loads(cleaned)
                
        except Exception as e:
            print(f"Error generating vocabulary with Ollama: {str(e)}")
            raise ValueError(f"Error generating vocabulary with Ollama: {str(e)}")

    def _clean_json_string(self, text: str) -> str:
        """Clean and extract JSON from the response text."""
        # Find the first '[' and last ']'
        start = text.find('[')
        end = text.rfind(']')
        
        if start == -1 or end == -1:
            raise ValueError("Could not find JSON array in response")
        
        # Extract just the array part
        json_text = text[start:end + 1]
        
        # Try to clean up any invalid JSON
        json_text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', json_text)
        json_text = re.sub(r'(?<!\\)\\\'', '\'', json_text)
        json_text = re.sub(r'(?<!\\)\\"', '"', json_text)
        json_text = re.sub(r'\'', '"', json_text)
        
        return json_text

def get_llm_provider() -> LLMProvider:
    """Get the first available LLM provider."""
    print("Trying to get LLM provider...")
    # Try Ollama first if available
    if OLLAMA_AVAILABLE:
        print("Ollama is available, trying to initialize...")
        try:
            return OllamaProvider()
        except Exception as e:
            print(f"Ollama error: {str(e)}")
    else:
        print("Ollama is not available")
    
    # Try Hugging Face if API key is available
    if HUGGINGFACE_AVAILABLE and os.getenv('HUGGINGFACE_API_KEY'):
        print("Hugging Face is available, trying to initialize...")
        try:
            return HuggingFaceProvider()
        except Exception as e:
            print(f"Hugging Face error: {str(e)}")
    else:
        print("Hugging Face is not available or API key is not set")
    
    raise ValueError("No LLM provider available. Please ensure either Hugging Face API key is set or Ollama is running locally.")

def generate_vocabulary(topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
    """Generate vocabulary using the first available LLM provider."""
    provider = get_llm_provider()
    return provider.generate_vocabulary(topic, language, num_words, difficulty, include_examples)
