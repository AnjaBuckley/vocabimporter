# VocabImporter

A Streamlit-based tool for generating and managing vocabulary for language learning applications. It uses Ollama's local LLM to generate vocabulary and can export directly to the Wortwunder backend.

## Features

- Generate vocabulary based on topics and difficulty levels
- Support for multiple languages (German, French, Spanish, Italian)
- Include example sentences for better context
- Export vocabulary to JSON files
- Import vocabulary from JSON files
- Direct integration with Wortwunder backend
- Docker support for easy deployment

## Prerequisites

- Docker and Docker Compose
- Ollama running locally with llama3.2 model installed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AnjaBuckley/vocabimporter.git
cd vocabimporter
```

2. Create a `.env` file:
```bash
OLLAMA_HOST=http://host.docker.internal:11434
WORTWUNDER_BACKEND_URL=http://localhost:5000
```

3. Build and run with Docker:
```bash
docker-compose up --build
```

4. Access the application at http://localhost:8501

## Usage

1. **Generate Vocabulary**:
   - Select target language
   - Enter topic/theme
   - Choose number of words
   - Set CEFR difficulty level
   - Toggle example sentences
   - Click "Generate Vocabulary"

2. **Export Options**:
   - Export to JSON file
   - Export directly to Wortwunder backend

3. **Import Vocabulary**:
   - Upload JSON file
   - Review imported vocabulary
   - Export to Wortwunder backend

## Docker Configuration

The application runs in a Docker container and connects to:
- Local Ollama service for LLM functionality
- Wortwunder backend for vocabulary storage

## Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
streamlit run app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
