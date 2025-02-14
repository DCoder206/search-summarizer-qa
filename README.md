# Web Search Query Summarizer & QA System

This project extracts meaningful text from Google search results, summarizes it using a Transformer-based model, and answers user queries based on the extracted information.

## Features
- **Search Engine Integration**: Fetches the top Google search results for a given query.
- **Text Extraction**: Strips unnecessary HTML tags and extracts relevant text.
- **Summarization**: Uses `facebook/bart-large-cnn` to condense extracted content.
- **Question Answering**: Utilizes `distilbert-base-cased-distilled-squad` to answer user queries based on the summarized text.
- **Chunking for Large Text**: Handles long texts by breaking them into manageable chunks before summarization.

## Installation
Ensure you have Python 3 installed, then install the required dependencies:

```bash
pip install transformers requests nltk
```

## Usage
Run the script and enter a search query when prompted:

```bash
python main.py
```

### Example
```bash
Enter your query: What is artificial intelligence?
```

### Expected Output
The script will:
1. Retrieve Google search results.
2. Extract and summarize relevant text.
3. Answer the query based on the extracted information.

```plaintext
Answer: Artificial intelligence (AI) refers to the simulation of human intelligence in machines.
```

## How It Works
1. **Fetching Search Results**: Constructs a Google search URL and fetches the HTML page.
2. **Text Extraction**: Uses regex to clean and extract readable text.
3. **Summarization**: Processes the extracted text using `facebook/bart-large-cnn`.
4. **Question Answering**: Uses `distilbert-base-cased-distilled-squad` to generate an answer based on the summarized content.

## Limitations
- Google may block automated requests; consider using official search APIs for large-scale use.
- Summarization is based on the extracted text and may not always retain all details.
- Responses depend on the quality of search results and model capabilities.

## Future Improvements
- Implement Reinforcement Learning (RLHF) to improve responses based on user feedback.
- Enhance HTML parsing using BeautifulSoup for better accuracy.
- Integrate more robust search APIs to avoid scraping limitations.

## License
This project is open-source under the MIT License.

