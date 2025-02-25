import json
import os
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

class LLMProvider(ABC):
    @abstractmethod
    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not installed")
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = openai.OpenAI(api_key=self.api_key)

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

    def _create_system_prompt(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> str:
        return f"""You are a language learning expert. Generate {num_words} {language} words related to '{topic}' 
        at {difficulty} level. For each word, provide:
        1. The word in {language}
        2. Its English translation
        3. Part of speech
        4. {"An example sentence" if include_examples else ""}
        
        Format the response EXACTLY as a JSON object with a 'words' key containing an array of word objects.
        Example format:
        {{
            "words": [
                {{
                    "word": "example_word",
                    "translation": "translation",
                    "part_of_speech": "noun",
                    "example_sentence": "This is an example."
                }}
            ]
        }}
        """

class OllamaProvider(LLMProvider):
    def __init__(self):
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        # Test connection to Ollama
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Could not connect to Ollama at {self.ollama_host}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.ollama_host}: {str(e)}")
        
        self.client = Client(base_url=self.ollama_host)

    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        system_prompt = self._create_system_prompt(topic, language, num_words, difficulty, include_examples)
        
        try:
            response = self.client.chat(
                model='llama3.2',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {num_words} {language} vocabulary words about {topic}"}
                ]
            )
            
            content = response['message']['content']
            cleaned_content = self._clean_json_response(content)
            vocab_data = json.loads(cleaned_content)
            return vocab_data.get('words', [])
        except Exception as e:
            raise Exception(f"Error generating vocabulary with Ollama: {str(e)}")

    def _clean_json_response(self, response_text: str) -> str:
        """Clean the response text by removing markdown backticks and finding the JSON content."""
        cleaned = response_text.strip('`')
        if cleaned.startswith('json'):
            cleaned = cleaned[4:]
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            return cleaned[start:end+1]
        return cleaned

    def _create_system_prompt(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> str:
        return f"""You are a language learning expert. Generate {num_words} {language} words related to '{topic}' 
        at {difficulty} level. For each word, provide:
        1. The word in {language}
        2. Its English translation
        3. Part of speech
        4. {"An example sentence" if include_examples else ""}
        
        Format the response EXACTLY as a JSON object with a 'words' key containing an array of word objects.
        Example format:
        {{
            "words": [
                {{
                    "word": "example_word",
                    "translation": "translation",
                    "part_of_speech": "noun",
                    "example_sentence": "This is an example."
                }}
            ]
        }}
        """

def get_llm_provider() -> LLMProvider:
    """Factory function to get the appropriate LLM provider"""
    errors = []
    
    # Check if we're running on Streamlit Cloud
    is_streamlit_cloud = os.getenv('STREAMLIT_RUNTIME_ENV') is not None or os.getenv('STREAMLIT_RUNTIME') is not None
    
    # If we're on Streamlit Cloud or have OpenAI key, try OpenAI first
    if (is_streamlit_cloud or os.getenv('OPENAI_API_KEY')) and OPENAI_AVAILABLE:
        try:
            return OpenAIProvider()
        except Exception as e:
            errors.append(f"OpenAI error: {str(e)}")
    
    # Only try Ollama if we're not on Streamlit Cloud
    if not is_streamlit_cloud:
        try:
            return OllamaProvider()
        except Exception as e:
            errors.append(f"Ollama error: {str(e)}")
    
    error_msg = "No LLM provider available:\n" + "\n".join(errors)
    raise Exception(error_msg)

def generate_vocabulary(topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
    """
    Generate vocabulary using the selected LLM provider.
    Returns a list of dictionaries containing word information.
    """
    provider = get_llm_provider()
    return provider.generate_vocabulary(topic, language, num_words, difficulty, include_examples)
