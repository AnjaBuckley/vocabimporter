import json
import os
import re
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import requests
from ollama import Client

# Try to import OpenAI, but don't fail if it's not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
            raise ImportError("Hugging Face packages are not installed")
        
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not self.api_key:
            raise ValueError("Hugging Face API key not found in environment variables")
        
        # Initialize client with Mistral model
        self.client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=self.api_key
        )
        print("Hugging Face client initialized successfully")

    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        system_prompt = self._create_system_prompt(topic, language, num_words, difficulty, include_examples)
        
        # Format prompt for Mistral
        prompt = f"<s>[INST]{system_prompt}[/INST]"
        
        # Generate response
        response = self.client.text_generation(
            prompt,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True
        )
        
        # Extract JSON from response
        try:
            # Find the first '[' and last ']' to extract the JSON array
            start = response.find('[')
            end = response.rfind(']') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = response[start:end]
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Raw response: {response}")
            raise ValueError("Failed to parse vocabulary from model response")

class OpenAIProvider(LLMProvider):
    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not installed")
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Debug: Print masked API key
        masked_key = f"{self.api_key[:8]}...{self.api_key[-4:]}" if self.api_key else "None"
        print(f"Initializing OpenAI client with key: {masked_key}")
        
        # Initialize client with explicit API key
        self.client = openai.OpenAI(
            api_key=self.api_key
        )
        print("OpenAI client initialized successfully")

    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        system_prompt = self._create_system_prompt(topic, language, num_words, difficulty, include_examples)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {num_words} {language} vocabulary words about {topic}"}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        vocab_data = json.loads(content)
        return vocab_data.get('words', [])

class OllamaProvider(LLMProvider):
    def __init__(self):
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama package is not installed")
        try:
            # Test connection
            ollama.ping(host=self.ollama_host)
        except Exception as e:
            raise ConnectionError(f"Could not connect to Ollama server at {self.ollama_host}: {str(e)}")

    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        system_prompt = self._create_system_prompt(topic, language, num_words, difficulty, include_examples)
        
        try:
            response = ollama.generate(
                model='mistral',
                prompt=system_prompt,
                host=self.ollama_host
            )
            
            # Get the response text
            response_text = response['response']
            print(f"Raw Ollama response: {response_text}")  # Debug print
            
            # Try to find and extract the JSON array
            try:
                # Find the first '[' and last ']' to extract the JSON array
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start == -1 or end == 0:
                    raise ValueError("No JSON array found in response")
                
                json_str = response_text[start:end]
                print(f"Extracted JSON: {json_str}")  # Debug print
                
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
                print(f"Cleaned JSON: {cleaned}")  # Debug print
                return json.loads(cleaned)
                
        except Exception as e:
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
        
        # Remove any trailing commas before closing brackets
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # Fix common formatting issues
        json_text = json_text.replace('\\n', ' ')
        json_text = json_text.replace('\n', ' ')
        json_text = re.sub(r'\s+', ' ', json_text)
        
        return json_text

def get_llm_provider() -> LLMProvider:
    """Get the first available LLM provider."""
    # Try Hugging Face first if API key is available
    if HUGGINGFACE_AVAILABLE and os.getenv('HUGGINGFACE_API_KEY'):
        try:
            return HuggingFaceProvider()
        except Exception as e:
            print(f"Hugging Face error: {str(e)}")
    
    # Try Ollama second if not on Streamlit Cloud
    if OLLAMA_AVAILABLE and not (os.getenv('STREAMLIT_RUNTIME_ENV') or os.getenv('STREAMLIT_RUNTIME')):
        try:
            return OllamaProvider()
        except Exception as e:
            print(f"Ollama error: {str(e)}")
    
    # Try OpenAI last
    if OPENAI_AVAILABLE:
        try:
            return OpenAIProvider()
        except Exception as e:
            print(f"OpenAI error: {str(e)}")
    
    raise ValueError("No LLM provider available. Please ensure either Hugging Face API key is set, Ollama is running locally, or OpenAI API key is configured.")

def generate_vocabulary(topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
    """
    Generate vocabulary using the selected LLM provider.
    Returns a list of dictionaries containing word information.
    """
    provider = get_llm_provider()
    return provider.generate_vocabulary(topic, language, num_words, difficulty, include_examples)
