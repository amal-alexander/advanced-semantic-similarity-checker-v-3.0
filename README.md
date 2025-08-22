# 🧠 Advanced Semantic Similarity Checker v3.0

A powerful Streamlit-based web application for advanced semantic similarity analysis across web pages. Leveraging state-of-the-art transformer models, multi-modal analytics, and robust content extraction, this tool enables deep comparison of website content for research, auditing, SEO, plagiarism detection, and more.

---

## 🚀 Features

- **Multi-Model Ensemble**: Utilizes several transformer models (MPNet, MiniLM, Paraphrase-MPNet) for robust similarity scoring.
- **Semantic Feature Analysis**: Extracts rich structural and vocabulary metrics for nuanced comparison.
- **Weighted Content Importance**: Prioritizes titles, headers, and meta descriptions to improve analysis accuracy.
- **Advanced Preprocessing**: Smart cleaning, normalization, and feature extraction for high-quality text data.
- **Error Recovery**: Multiple retry attempts and clear reporting for unreliable web sources.
- **Detailed Analytics**: Provides both page-level and block-level similarity breakdowns.
- **Flexible Input**: Accepts URLs via direct input or file upload (.txt, .csv, .xlsx).
- **Customizable UI**: Easily configure analysis depth, similarity thresholds, and comparison modes.
- **Export Options**: Download results as CSV or detailed JSON reports.

---

## ⚡️ Quick Start

1. **Install dependencies:**
    ```bash
    pip install streamlit pandas numpy sentence-transformers beautifulsoup4 requests scikit-learn nltk
    ```
    > If you plan to use `.xlsx` uploads, also install:
    ```bash
    pip install openpyxl
    ```

2. **Run the app:**
    ```bash
    streamlit run app.py
    ```

3. **Navigate to the UI:**  
    The app will open in your browser. Paste URLs or upload a file, configure your analysis, and start!

---

## 🛠️ Usage Guide

### Input Methods

- **Paste URLs**: Enter one URL per line in the text area.
- **Upload File**: Supports `.txt`, `.csv`, or `.xlsx` files with URLs in the first column.

### Analysis Configuration

- **Comparison Modes**:  
  - *All vs All*: Full matrix (for ≤10 URLs)
  - *Sequential Pairs*: Adjacent pairs (for bulk)
  - *One vs Rest*: Compare first URL to all others

- **Detail Level**:  
  - *Page-Level Only*: Fast summary
  - *Block-Level Analysis*: Fine-grained comparison of content blocks
  - *Full Deep Analysis*: Most detailed (may be slow for large sets)

- **Thresholds & Advanced Settings**:  
  Adjust similarity thresholds, max URLs, and request timeouts via the sidebar.

### Results

- **Overview Table**: See status, scores, and key metrics for each pair.
- **Detailed Block Analysis**: Drill down to highly similar content blocks between pages.
- **Export**: Download CSV/JSON reports for further use.

---

## 🧑‍💻 Developer Notes

- **Model Loading**: Efficient caching and fallback handling for transformer models.
- **NLTK Setup**: Automatic downloads for required corpora.
- **Error Handling**: Graceful error messages for content extraction and analysis.
- **Customization**: UI and logic are modular for easy extension.

---

## 🤝 Attribution

Developed by [Amal Alexander](https://www.linkedin.com/in/amal-alexander-305780131/).

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## ⭐️ Support & Contact

amalalex95@gmail.com

For issues, suggestions, or collaboration, feel free to open a GitHub issue or reach out via [LinkedIn](https://www.linkedin.com/in/amal-alexander-305780131/).
