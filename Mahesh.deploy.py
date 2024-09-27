#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

book= pd.read_csv(r"C:\Users\harik\OneDrive\Pictures\Downloads\Books.csv",low_memory=False)


# In[69]:


# Assuming 'final_ratings.csv' is loaded somewhere in the code
ratings_data= pd.read_csv(r"C:\Users\harik\OneDrive\Pictures\Downloads\Ratings.csv")


# In[70]:


ratings_with_name = ratings_data.merge(book,on='ISBN')


# In[71]:


x=ratings_with_name.groupby('User-ID').count()['Book-Rating'] >200
users=x[x].index


# In[72]:


filtered_rating=ratings_with_name[ratings_with_name['User-ID'].isin(users)]


# In[73]:


y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index


# In[74]:


final_rating= filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
final_rating.drop_duplicates()


# In[75]:


pt =final_rating.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)


# In[76]:


st.title(' Book Recommendation System by User-Based ')

# Function to recommend similar books
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]

    similarity_scores = cosine_similarity(pt)
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

    recommended_books = [pt.index[i[0]] for i in similar_items]
    return recommended_books

# Streamlit app
def main():
    st.title("Recommand Top 5 Books")
    
    user_input = st.text_input("Enter User ID or Book Title:")

    if st.button("Enter"):
        try:
            user_input = int(user_input)
            if user_input in pt.columns:
                user_ratings = pt[user_input].dropna()
                top_rated_books = user_ratings.sort_values(ascending=False).head(5)

                st.write(f"Top 5 rated books for User {user_input}:")
                st.write(top_rated_books)
            else:
                st.write("Invalid User ID. Please enter a valid User ID.")
        except ValueError:
            if user_input in pt.index:
                recommended_books = recommend(user_input)
                st.write(f"Books similar to '{user_input}':")
                st.write(recommended_books)
            else:
                st.write(f"Book '{user_input}' not found in the dataset.")

    st.title("Recommand Books")

    book_input = st.text_input("Enter Book Name:")

    if st.button("Recommend"):
        if book_input in pt.index:
            recommended_books = recommend(book_input)
            st.write(f"Books similar to '{book_input}':")
            st.write(recommended_books)
        else:
            st.write(f"Book '{book_input}' not found in the dataset.")

if __name__ == "__main__":
    main()

