import pickle
import streamlit as st
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.header('Book Recommender System Using Machine Learning')
model = pickle.load(open('model.pkl','rb'))
book_names = pickle.load(open('book_names.pkl','rb'))
final_rating = pickle.load(open('filtered_rating.pkl','rb'))
book_pivot = pickle.load(open('pt.pkl','rb'))


def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    #print(book_name)
    for name in book_name: 
        ids = np.where(final_rating['Book-Title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['Image-URL-M']
        poster_url.append(url)

    return poster_url



def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    #print(book_id)
    distance, suggestions = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )
    suggestions=suggestions.flatten()[1:]
    poster_url = fetch_poster(suggestions)
    #print(poster_url)
    for i in range(len(suggestions)):
            book = book_pivot.index[suggestions[i]]
            books_list.append(book)
    return books_list,poster_url     



selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)
if st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True):

    if st.button('Show Recommendation'):
        # pass
        recommended_books,poster_url = recommend_book(selected_books)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_books[0])
            st.image(poster_url[0])
        with col2:
            st.text(recommended_books[1])
            st.image(poster_url[1])

        with col3:
            st.text(recommended_books[2])
            st.image(poster_url[2])
        with col4:
            st.text(recommended_books[3])
            st.image(poster_url[3])
        with col5:
            st.text(recommended_books[4])
            st.image(poster_url[4])

# book_name = "Harry Potter and the Chamber of Secrets (Book 2)"
# print('book name : ',book_name)
# #print(len(book_pivot.index.tolist()))
# print("Recommended Books:")
# print(recommend_book(book_name))