from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import warnings
import numpy as np
import pandas as pd
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')


app = FastAPI()

model = joblib.load(r"D:\Machine_learning\Deeplearning\aiml_tutorials\recommendations_systems\campusx\book-recommender-system\model.pkl")
book_names = joblib.load(r"D:\Machine_learning\Deeplearning\aiml_tutorials\recommendations_systems\campusx\book-recommender-system\book_names.pkl")
final_rating = joblib.load(r"D:\Machine_learning\Deeplearning\aiml_tutorials\recommendations_systems\campusx\book-recommender-system\filtered_rating.pkl")
book_pivot = joblib.load(r"D:\Machine_learning\Deeplearning\aiml_tutorials\recommendations_systems\campusx\book-recommender-system\pt.pkl")


MODEL_VERSION='1.0.1'

# Define request body
class RecommendationRequest(BaseModel):
    book_name: str

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

# human readable       
@app.get('/')
def home():
    return {'message':'Book Recommendations '}

# machine readable
@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'version': MODEL_VERSION,
        'model_loaded': model is not None
    }

@app.post('/recommendations')
def get_recommendations(request: RecommendationRequest):
    try:
        books, posters = recommend_book(request.book_name)
        return JSONResponse(status_code=200, content={'books': books, 'posters': posters})
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))





