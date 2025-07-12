# 📚 Book Recommender System using Machine Learning, FastAPI & Streamlit

This is a content-based book recommendation system built with **Scikit-learn**, **FastAPI**, and **Streamlit**. The model suggests similar books based on user input and displays book covers and titles.

---

## 🚀 Features

- 🔍 Real-time book recommendations via FastAPI
- 🖼️ Shows book titles and cover images
- 🌐 Web frontend using Streamlit
- ⚡ Fast and interactive UI
- 🔗 API endpoint to integrate with any frontend

---

## 📦 Requirements

Before running the app, install the required dependencies:

```bash
pip install -r requirements.txt


Then run the api:

```bash
uvicorn model:app --reload

Finally run frontend app:

```bash
streamlit run app.py
