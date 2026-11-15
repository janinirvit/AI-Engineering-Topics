import streamlit as st
import os
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
from news_summarizer import NewsArticleSummarizer

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Article Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, professional styling
st.markdown("""
    <style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        padding: 30px 20px;
        margin-bottom: 10px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        color: #555;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Input section styling */
    .input-section {
        background: white;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e7ff;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 22px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    /* Info labels */
    .info-label {
        font-weight: 700;
        color: #667eea;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        display: inline-block;
    }
    
    .info-value {
        color: #2d3748;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0;
    }
    
    /* Summary section */
    .summary-section {
        background: white;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        border-left: 5px solid #f093fb;
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.1);
    }
    
    .summary-title {
        color: #f093fb;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .summary-content {
        color: #2d3748;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    
    /* Document content */
    .doc-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #e0e7ff;
        margin: 12px 0;
        max-height: 350px;
        overflow-y: auto;
        color: #2d3748;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8e9f3 100%);
    }
    
    .sidebar-title {
        color: #667eea;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 15px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 40px;
        font-weight: 700;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Success message */
    .success-msg {
        background: linear-gradient(135deg, #d4fc79 0%, #96f550 100%);
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 8px;
        color: #2d3748;
        font-weight: 600;
    }
    
    /* Error message */
    .error-msg {
        background: linear-gradient(135deg, #ff9a56 0%, #ff7e5f 100%);
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    /* Divider styling */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 30px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.85rem;
        margin-top: 40px;
        padding-top: 30px;
        border-top: 2px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Metadata grid */
    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin: 20px 0;
    }
    
    @media (max-width: 768px) {
        .metadata-grid {
            grid-template-columns: 1fr;
        }
        .main-header {
            font-size: 2rem;
        }
    }
    
    /* Scroll styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
    """, unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">📰 Article Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Extract key insights from any article using AI-powered summarization</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Configuration</div>', unsafe_allow_html=True)
    st.divider()
    
    # Model selection
    model_name = st.selectbox(
        "🤖 Select Model:",
        ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Choose the OpenAI model for summarization"
    )
    
    # Summary type selection
    summary_type = st.radio(
        "📝 Summary Type:",
        ["detailed", "concise"],
        help="Choose between detailed or concise summary"
    )
    
    st.divider()
    st.markdown("### 💡 How to Use")
    st.markdown("""
    1. **Paste Article URL** - Any publicly accessible article
    2. **Select Model** - Choose processing speed vs quality
    3. **Choose Summary Type** - Detailed or concise
    4. **Click Summarize** - Wait for AI to process
    5. **Review & Download** - Save your summary
    """)
    
    st.divider()
    st.markdown("### 📌 About This Tool")
    st.markdown("""
    **Powered by:**
    - LangChain - Document orchestration
    - OpenAI GPT - Advanced language model
    - Streamlit - Beautiful UI
    
    **Supports:**
    - News articles
    - Blog posts
    - Research papers
    - Technical articles
    """)

st.divider()

# Input section with better styling
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    article_url = st.text_input(
        "📎 Enter Article URL:",
        placeholder="Paste the full URL of the article you want to summarize...",
        help="Example: https://example.com/article-title"
    )

with col2:
    summarize_button = st.button("🚀 Summarize", use_container_width=True, type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Processing section
if summarize_button:
    if not article_url:
        st.markdown('<div class="error-msg">❌ Please enter a valid article URL</div>', unsafe_allow_html=True)
    elif not article_url.startswith(("http://", "https://")):
        st.markdown('<div class="error-msg">❌ Please enter a URL starting with http:// or https://</div>', unsafe_allow_html=True)
    else:
        try:
            with st.spinner("🔄 Fetching and processing article... This may take a moment."):
                # Initialize summarizer
                summarizer = NewsArticleSummarizer(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_type="openai",
                    model_name=model_name
                )
                
                # Summarize article
                result = summarizer.summarize_article(article_url, summary_type=summary_type)
                
                if "error" in result:
                    st.markdown(f'<div class="error-msg">❌ Error: {result["error"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-msg">✅ Article summarized successfully!</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    
                    # Display metadata in grid layout
                    st.markdown("### 📋 Article Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown('<p class="info-label">📑 Title</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="info-value">{result["title"]}</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown('<p class="info-label">📅 Published Date</p>', unsafe_allow_html=True)
                        if result['publish_date']:
                            st.markdown(f'<p class="info-value">{str(result["publish_date"])}</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="info-value">Date not available</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown('<p class="info-label">🤖 Model Used</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="info-value">{result["model_info"]["type"].upper()} - {result["model_info"]["name"]}</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown('<p class="info-label">✍️ Authors</p>', unsafe_allow_html=True)
                        if result['authors']:
                            st.markdown(f'<p class="info-value">{", ".join(result["authors"])}</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="info-value">Authors not available</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    
                    # Summary section
                    st.markdown("### 📝 AI-Generated Summary")
                    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
                    
                    # Extract summary text
                    summary_text = result['summary'].get('output_text', str(result['summary']))
                    st.markdown(f'<div class="summary-content">{summary_text}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    
                    # Document content sections
                    st.markdown("### 📄 Original Document Chunks")
                    
                    # Get input documents
                    input_docs = result['summary'].get('input_documents', [])
                    
                    if len(input_docs) > 0:
                        with st.expander("📄 First Document Chunk", expanded=False):
                            st.markdown('<div class="doc-content">', unsafe_allow_html=True)
                            st.write(input_docs[0].page_content)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    if len(input_docs) > 1:
                        with st.expander("📄 Second Document Chunk", expanded=False):
                            st.markdown('<div class="doc-content">', unsafe_allow_html=True)
                            st.write(input_docs[1].page_content)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    if len(input_docs) > 2:
                        with st.expander(f"📄 Additional Chunks ({len(input_docs) - 2} more)", expanded=False):
                            for idx, doc in enumerate(input_docs[2:], start=3):
                                with st.expander(f"Chunk {idx}", expanded=False):
                                    st.markdown('<div class="doc-content">', unsafe_allow_html=True)
                                    st.write(doc.page_content)
                                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    
                    # Download and sharing options
                    st.markdown("### 📥 Export & Share")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Create formatted summary text for download
                    export_text = f"""╔═══════════════════════════════════════════════════════════╗
║           ARTICLE SUMMARY REPORT                           ║
╚═══════════════════════════════════════════════════════════╝

📑 TITLE
{result['title']}

📅 PUBLICATION DATE
{result['publish_date'] if result['publish_date'] else 'Not available'}

✍️ AUTHORS
{', '.join(result['authors']) if result['authors'] else 'Not available'}

🤖 AI MODEL USED
{result['model_info']['type'].upper()} - {result['model_info']['name']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 SUMMARY
{summary_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔗 SOURCE URL
{article_url}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated using Article Summarizer with LangChain & OpenAI
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    
                    with col1:
                        st.download_button(
                            label="📄 Download Summary",
                            data=export_text,
                            file_name="article_summary.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        st.info("💡 Tip: Copy the summary and share on LinkedIn, email, or documents")
                    
        
        except Exception as e:
            st.markdown(f'<div class="error-msg">❌ An error occurred: {str(e)}</div>', unsafe_allow_html=True)
            st.info("💡 Tip: Make sure your OpenAI API key is set correctly in the .env file")

# Footer
st.markdown("""
<div class="footer">
    <p>🚀 Built with Streamlit | Powered by LangChain & OpenAI</p>
    <p>Save time, extract insights, stay informed.</p>
</div>
""", unsafe_allow_html=True)
