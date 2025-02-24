import json
import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
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
        openai.api_key = self.api_key

    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        system_prompt = self._create_system_prompt(topic, language, num_words, difficulty, include_examples)
        
        response = openai.ChatCompletion.create(
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
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.client = Client(base_url=ollama_host)

    def generate_vocabulary(self, topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
        system_prompt = self._create_system_prompt(topic, language, num_words, difficulty, include_examples)
        
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
    # Try Ollama first
    try:
        return OllamaProvider()
    except Exception as e:
        print(f"Failed to initialize Ollama provider: {e}")
    
    # Fall back to OpenAI if available and configured
    if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
        try:
            return OpenAIProvider()
        except Exception as e:
            print(f"Failed to initialize OpenAI provider: {e}")
    
    raise Exception("No LLM provider available. Please ensure either Ollama is running or OpenAI API key is set.")

def generate_vocabulary(topic: str, language: str, num_words: int, difficulty: str, include_examples: bool) -> List[Dict]:
    """
    Generate vocabulary using the selected LLM provider.
    Returns a list of dictionaries containing word information.
    """
    provider = get_llm_provider()
    return provider.generate_vocabulary(topic, language, num_words, difficulty, include_examples)
