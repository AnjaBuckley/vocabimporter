# VocabImporter

A Streamlit-based tool for generating and managing vocabulary for language learning applications. It uses OpenAI or Ollama's local LLM to generate vocabulary and can export directly to the Wortwunder backend.

## Features

- Generate vocabulary based on topics and difficulty levels
- Support for multiple languages (German, French, Spanish, Italian)
- Include example sentences for better context
- Export vocabulary to JSON files
- Import vocabulary from JSON files
- Direct integration with Wortwunder backend
- Support for both OpenAI and local Ollama LLMs

## Prerequisites

For local development:
- Docker and Docker Compose (if using Ollama)
- Ollama running locally with llama3.2 model installed (optional)
- OpenAI API key (optional)

For cloud deployment:
- OpenAI API key
- Streamlit Cloud account

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/AnjaBuckley/vocabimporter.git
cd vocabimporter
```

2. Create a `.env` file:
```bash
# If using Ollama
OLLAMA_HOST=http://host.docker.internal:11434

# If using OpenAI
OPENAI_API_KEY=your_api_key_here

# Backend configuration
WORTWUNDER_BACKEND_URL=http://localhost:5000
```

3. Build and run with Docker:
```bash
docker-compose up --build
```

4. Access the application at http://localhost:8501

### Cloud Deployment

1. Fork this repository to your GitHub account

2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)

3. Create a new app in Streamlit Cloud:
   - Connect your GitHub account
   - Select the forked repository
   - Select the main branch
   - Set the path to `app.py`

4. Add the following secrets in Streamlit Cloud settings:
   ```
   OPENAI_API_KEY=your_api_key_here
   WORTWUNDER_BACKEND_URL=your_backend_url
   ```

5. Deploy! Streamlit Cloud will automatically build and deploy your app

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

## LLM Provider Configuration

The app supports two LLM providers:

1. **OpenAI (Recommended for cloud deployment)**
   - Set `OPENAI_API_KEY` environment variable
   - Uses GPT-3.5-turbo model
   - Better suited for cloud deployment

2. **Ollama (Local development)**
   - Set `OLLAMA_HOST` environment variable
   - Requires local Ollama installation
   - Uses llama3.2 model
   - Better for offline development

The app will automatically use OpenAI if an API key is provided, otherwise it will fall back to Ollama.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
