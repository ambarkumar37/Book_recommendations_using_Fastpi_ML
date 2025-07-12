import pickle
import streamlit as st
import requests  # <-- Needed for HTTP POST
import warnings
warnings.filterwarnings('ignore')

API_URL = "http://127.0.0.1:8000/recommendations"  # FastAPI endpoint

st.header('📚 Book Recommender System Using Machine Learning')

# Load book names for dropdown
book_names = pickle.load(open('book_names.pkl', 'rb'))

# Dropdown for book selection
selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

# Green button styling
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Show Recommendations
if st.button('Show Recommendation'):
    try:
        # Make API call
        response = requests.post(API_URL, json={"book_name": selected_books})

        if response.status_code == 200:
            result = response.json()
            recommended_books = result['books']
            poster_url = result['posters']

            # Show 5 columns of recommendations
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.text(recommended_books[i])
                    st.image(poster_url[i])
        else:
            st.error(f"Error from server: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"API call failed: {e}")
