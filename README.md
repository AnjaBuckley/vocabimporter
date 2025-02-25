# VocabImporter

A Streamlit-based tool for generating and managing vocabulary for language learning applications. It uses Hugging Face's Mixtral model or Ollama's local LLM to generate vocabulary and can export directly to the Wortwunder backend.

## Features

- Generate vocabulary based on topics and difficulty levels
- Support for multiple languages (German, French, Spanish, Italian)
- Include example sentences for better context
- Export vocabulary to JSON files
- Import vocabulary from JSON files
- Direct integration with Wortwunder backend
- Support for both Hugging Face and local Ollama LLMs

## Prerequisites

For local development:
- Docker and Docker Compose
- Ollama running locally with llama2 model installed (optional)
- Hugging Face API key (optional)

For cloud deployment:
- Hugging Face API key
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
# If using Ollama (fallback option)
OLLAMA_HOST=http://ollama:11434

# If using Hugging Face (recommended)
HUGGINGFACE_API_KEY=your_api_key_here

# Backend configuration
WORTWUNDER_BACKEND_URL=http://localhost:5000
```

3. Build and run with Docker:
```bash
docker-compose up --build
```

4. Access the application at http://localhost:8501

### Cloud Deployment (Streamlit Cloud)

1. Fork this repository to your GitHub account

2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)

3. Create a new app in Streamlit Cloud:
   - Connect your GitHub account
   - Select the forked repository
   - Select the main branch
   - Set the path to `app.py`

4. Configure Streamlit Cloud secrets:
   - Click on "Manage app" in the bottom right
   - Go to the "Secrets" section
   - Click "Edit Secrets"
   - Add the following in TOML format:
   ```toml
   HUGGINGFACE_API_KEY = "your_huggingface_api_key"
   ```
   - Note: Do NOT add OLLAMA_HOST for cloud deployment

5. Deploy:
   - The app will automatically deploy
   - If you make changes to your GitHub repository, Streamlit Cloud will automatically redeploy

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

1. **Hugging Face (Recommended)**
   - Set `HUGGINGFACE_API_KEY` environment variable or Streamlit secret
   - Uses Mixtral model
   - Better suited for both cloud and local deployment
   - Higher quality vocabulary generation

2. **Ollama (Local development only)**
   - Set `OLLAMA_HOST` environment variable (local only)
   - Uses local Llama2 model
   - Good for local development and testing
   - No API key required
   - Requires more computational resources

The app will automatically use Hugging Face if the API key is available, otherwise it will fall back to using Ollama (local development only).

## Troubleshooting

### Common Issues

1. **Streamlit Cloud Deployment**
   - Make sure you've added the `HUGGINGFACE_API_KEY` in Streamlit Cloud secrets
   - The app will not work with Ollama in cloud deployment
   - Check the app logs in Streamlit Cloud for detailed error messages

2. **Local Development**
   - Ensure Docker and Docker Compose are installed
   - Check that Ollama is running if you're not using Hugging Face
   - Verify your `.env` file exists and contains the correct variables

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License
