# 🎯 YouTube Comment Sentiment Analysis

This project analyzes YouTube video comments and classifies them into **positive**, **neutral**, or **negative** sentiments using a multilingual BERT-based transformer model.

---

## 📌 Features

- 🔍 **YouTube Data API Integration** to fetch real-time comments  
- 🤖 **Multilingual sentiment classification** using Hugging Face Transformers (`distilbert-base-multilingual-cased`)  
- 🧹 **Text preprocessing** (cleaning, lowercasing, removing URLs/symbols)  
- 📊 **Pie chart visualization** of sentiment distribution  
- 💬 Works on **any YouTube video** with comments enabled

---

## 🚀 Demo

![Sentiment Pie Chart](./example_output.png)

---

## 📦 Technologies Used

- `Python`
- `transformers` (Hugging Face)
- `torch`
- `pandas`, `numpy`
- `matplotlib`
- `google-api-python-client`
- `nltk` (for text cleaning)

---

## 📥 Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/dhawanaaaryan/YoutubeCommentAnalysis.git
   cd YoutubeCommentAnalysis
