import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st
from data import data

df = pd.DataFrame(data, columns=["ID", "Name", "Interests", "University", "Citations"])

df["Profile"] = df["Name"] + " " + df["Interests"] + " " + df["University"]

model = SentenceTransformer('all-MiniLM-L6-v2')

df["Embedding"] = list(model.encode(df["Profile"]))

user_name = "Fahim"
user_interest = "AI and Data Science, Machine Learning, Robotics", "Bio informatics"
user_university = "MIT"

def get_recommendations(user_name, user_interest, user_university):
    user_profile = f"{user_name} {user_interest} {user_university}"
    user_embedding = model.encode([user_profile])
    user_embedding = user_embedding.reshape(1, -1)
    df_embeddings = list(df["Embedding"])
    similarities= cosine_similarity(user_embedding, df_embeddings)
    df["Similarity"] = similarities.flatten()
    recommended = df.sort_values(by="Similarity", ascending=False)
    top_3_recommendations = recommended[["ID", "Name", "Interests", "University", "Citations"]].head(3)
    return top_3_recommendations


st.title("Researcher Recommendation System")

user_name = st.text_input("Enter your name")
user_interest = st.text_area("Enter your research interests")
user_university = st.text_input("Enter your university")

st.markdown("""
    <style>
        .dataframe {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto;
        }
        .stTable th, .stTable td {
            text-align: center;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

if st.button("Get Recommedation"):
    if user_name and user_interest and user_university:
        recommendations = get_recommendations(user_name, user_interest, user_university)
        st.table(recommendations)
    else:
        st.warning("Please fill in all the fields")


