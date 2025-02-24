import json
import os
from ollama import Client

# Initialize the Ollama client with base_url from environment variable
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
client = Client(base_url=ollama_host)

def clean_json_response(response_text):
    """Clean the response text by removing markdown backticks and finding the JSON content."""
    # Remove markdown code block if present
    cleaned = response_text.strip('`')
    if cleaned.startswith('json'):
        cleaned = cleaned[4:]  # Remove 'json' from the start
    # Find the first '{' and last '}'
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1:
        return cleaned[start:end+1]
    return cleaned

def generate_vocabulary(topic, language, num_words, difficulty, include_examples=True):
    """
    Generate vocabulary using Ollama.
    Returns a list of dictionaries containing word information.
    """
    system_prompt = f"""You are a language learning expert. Generate {num_words} {language} words related to '{topic}' 
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

    try:
        response = client.chat(
            model='llama3.2',  # Using the installed llama3.2 model
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Generate {num_words} {language} vocabulary words about {topic}"
                }
            ]
        )
        
        # Extract and process the JSON response
        try:
            content = response['message']['content']
            cleaned_content = clean_json_response(content)
            vocab_data = json.loads(cleaned_content)
            return vocab_data.get('words', [])
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {content}")
        
    except Exception as e:
        raise Exception(f"Error generating vocabulary: {str(e)}")
