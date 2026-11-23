# 📰 News Summarizer Module

A complete AI-powered news article summarization system using LangChain and OpenAI, part of the `langchain_apps/` folder.

## 📁 Folder Structure

```
langchain_apps/
├── README.md                           # This file - Overview and guide
├── ARTICLE_SUMMARIZER_README.md        # Detailed feature documentation
├── STREAMLIT_SETUP_GUIDE.md            # Setup and configuration guide
├── QUICK_START.md                      # 5-minute quick start guide
│
├── Core Files
├── news_summarizer.py                  # Core NewsArticleSummarizer class
├── streamlit_article_summarizer.py     # Streamlit web UI application
├── test_chain_types.py                 # Unit tests for chain types
├── youtube_video_summarizer.py         # YouTube video summarizer app
```

## 🚀 Quick Start (30 seconds)

### 1. Install Dependencies
```bash
pip install -r ../requirements.txt
```

### 2. Set API Key
Create `.env` in project root:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the App
```bash
cd langchain_apps
streamlit run streamlit_article_summarizer.py
```

**✅ Done!** App opens at `http://localhost:8501`

## 📚 Files Overview

### Core Files

#### **news_summarizer.py** - Core Class
The heart of the system. Contains the `NewsArticleSummarizer` class that handles:
- Article fetching from URLs using Newspaper3k
- Text splitting with configurable chunk sizes
- Summarization using multiple chain types (stuff, map_reduce, refine)
- Support for different LLMs (OpenAI, Ollama)

**Key Methods:**
- `fetch_article(url)` - Downloads and parses articles
- `summarize_article(url, summary_type)` - Main summarization method
- `_validate_chain_type(chain_type)` - Validates chain type parameter

#### **streamlit_article_summarizer.py** - Web UI
Beautiful Streamlit web interface for the summarizer with:
- Real-time article summarization
- Model and chain type selection
- Detailed/concise summary options
- Document chunk viewing
- Export to text file
- Professional gradient UI with custom CSS

#### **test_chain_types.py** - Unit Tests
Comprehensive test suite covering:
- Chain type validation
- Parameter storage
- Case insensitivity
- Invalid input rejection
- Return dictionary structure

**Run tests:**
```bash
python test_chain_types.py
```

### Documentation Files

#### **ARTICLE_SUMMARIZER_README.md** - Feature Documentation
Complete documentation including:
- Architecture overview
- Component descriptions
- Configuration options
- Advanced usage examples
- Troubleshooting guide

#### **STREAMLIT_SETUP_GUIDE.md** - Setup Guide
Detailed setup instructions:
- Environment configuration
- Dependency installation
- Running the app
- Configuration options
- Troubleshooting
- Performance benchmarks

#### **QUICK_START.md** - Quick Reference
Fast 5-minute guide:
- 30-second setup
- Basic usage
- Chain types explained
- Example URLs
- Testing

## 🎯 Supported Chain Types

| Type | Speed | Quality | Use Case |
|------|-------|---------|----------|
| **stuff** | ⚡⚡⚡ | ⭐⭐ | Short articles (<3000 tokens) |
| **map_reduce** | ⚡⚡ | ⭐⭐⭐ | Long articles (balanced) |
| **refine** | ⚡ | ⭐⭐⭐⭐ | Maximum quality summarization |

## 🤖 Supported Models

- **GPT-4** - Most capable, best quality
- **GPT-4o-mini** - Fast, cost-effective (recommended)
- **GPT-3.5-turbo** - Budget-friendly option
- **Llama3.2** (via Ollama) - Local option

## 📖 Usage Examples

### Basic Python Usage
```python
from news_summarizer import NewsArticleSummarizer

# Initialize
summarizer = NewsArticleSummarizer(
    model_type="openai",
    model_name="gpt-4o-mini",
    chain_type="map_reduce"
)

# Summarize
result = summarizer.summarize_article(
    url="https://example.com/article",
    summary_type="detailed"
)

print(f"Title: {result['title']}")
print(f"Summary: {result['summary']}")
```

### Web App Usage
```bash
streamlit run streamlit_article_summarizer.py
```
Then:
1. Paste article URL
2. Select model and chain type
3. Click "Summarize"
4. Download results (optional)

## ⚙️ Configuration

### Environment Variables (.env)
```
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
OPENAI_MODEL=gpt-4o-mini
CHAIN_TYPE=map_reduce
ARTICLE_CHUNK_SIZE=2000
ARTICLE_CHUNK_OVERLAP=200
```

### Python Configuration
```python
# Default settings
summarizer = NewsArticleSummarizer(
    api_key=None,                    # Uses OPENAI_API_KEY env var
    model_type="openai",             # "openai" or "ollama"
    model_name="gpt-4o-mini",        # Model name
    chain_type="map_reduce"          # "stuff", "map_reduce", "refine"
)
```

## 🧪 Testing

### Run Unit Tests
```bash
python test_chain_types.py
```

Expected output:
```
TEST 1: Chain Type Validation ✓
TEST 2: Chain Type Parameter Storage ✓
TEST 3: VALID_CHAIN_TYPES Constant ✓
TEST 4: Return Dictionary Includes chain_type ✓

ALL 4 TESTS PASSED!
```

### Manual Testing
1. Start the Streamlit app
2. Test with different article URLs
3. Try each chain type
4. Verify model selection works
5. Test download functionality

## 📊 Output Format

The `summarize_article()` method returns a dictionary with:
```python
{
    "title": str,              # Article title
    "authors": list,           # List of authors
    "publish_date": datetime,  # Publication date
    "summary": dict,           # LangChain summary result
    "url": str,                # Source URL
    "model_info": dict,        # Model used (type, name)
    "chain_type": str,         # Chain type used
}
```

## 🔧 Troubleshooting

### "API Key Error"
- Verify `.env` file exists in project root
- Check `OPENAI_API_KEY` is correct
- Restart Streamlit after updating `.env`

### "Failed to fetch article"
- Verify URL is accessible in browser
- Some websites block automated access
- Try a different article URL

### "Module not found"
```bash
pip install -r ../requirements.txt
```

### "Chain type validation error"
- Use only: `stuff`, `map_reduce`, or `refine`
- Input is case-insensitive (converts to lowercase)

## 📈 Performance Tips

1. **Faster Processing:**
   - Use `gpt-4o-mini` model
   - Select `concise` summary type
   - Use `stuff` chain for short articles

2. **Better Quality:**
   - Use `gpt-4` model
   - Select `detailed` summary type
   - Use `refine` chain type

3. **Cost Optimization:**
   - Use `gpt-4o-mini` for most tasks
   - Use `gpt-3.5-turbo` only for budget constraints
   - Estimate: 500 articles/month = $5-100 depending on model

## 🚀 Deployment Options

### Local Streamlit (Development)
```bash
streamlit run streamlit_article_summarizer.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Create account at streamlit.io
3. Deploy from GitHub repo
4. Add secrets for API keys

### Docker Container
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY langchain_apps/ .
CMD ["streamlit", "run", "streamlit_article_summarizer.py"]
```

## 📝 Development

### Adding New Features

1. **New Chain Type Support:**
   - Add to `VALID_CHAIN_TYPES` in `news_summarizer.py`
   - Implement prompt configuration in `_load_chain()`
   - Add test case in `test_chain_types.py`

2. **New Model Support:**
   - Add to model initialization in `__init__()`
   - Update error handling
   - Test with existing articles

3. **UI Customization:**
   - Modify CSS in `streamlit_article_summarizer.py`
   - Add new sidebar options
   - Enhance result display

### Contributing

1. Test changes locally
2. Update relevant documentation
3. Run test suite: `python test_chain_types.py`
4. Commit with clear messages

## 📚 Additional Resources

- **LangChain Docs:** https://python.langchain.com
- **Streamlit Docs:** https://docs.streamlit.io
- **OpenAI API:** https://platform.openai.com/docs
- **Newspaper3k:** https://newspaper.readthedocs.io

## 📋 Next Steps

1. ✅ Read QUICK_START.md for immediate setup
2. ✅ Run the Streamlit app
3. ✅ Test with different articles
4. ✅ Try each chain type and model
5. ✅ Review ARTICLE_SUMMARIZER_README.md for advanced features
6. ✅ Customize and extend as needed

## 📄 License

Part of AI-Engineering-Topics repository

---

**Happy Summarizing! 🎉**

For detailed setup instructions, see [STREAMLIT_SETUP_GUIDE.md](STREAMLIT_SETUP_GUIDE.md)
For quick 5-minute start, see [QUICK_START.md](QUICK_START.md)
For feature documentation, see [ARTICLE_SUMMARIZER_README.md](ARTICLE_SUMMARIZER_README.md)
