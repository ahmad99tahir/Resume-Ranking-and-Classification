import pandas as pd
import re
from src.preprocessing import clean_resume
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(file_path):
    """
    Preprocesses the data by removing duplicates, resetting the index, and cleaning resumes.

    Args:
    file_path (str): Path to the CSV file containing resume data.

    Returns:
    pandas.DataFrame: Processed DataFrame with cleaned resumes.
    """
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['cleaned_resume'] = df['Resume'].apply(lambda x: clean_resume(x))
    return df

def compute_similarity(documents, keywords):
    """
    Computes cosine similarity between documents and a set of keywords.

    Args:
    documents (list): List of document strings.
    keywords (list): List of keyword strings.

    Returns:
    numpy.ndarray: Array containing cosine similarity scores.
    """
    all_documents = keywords + documents.tolist()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[:len(keywords)], tfidf_matrix[len(keywords):])
    return cosine_similarities

def sort_resumes_by_similarity(documents, cosine_similarities):
    """
    Sorts resumes by average cosine similarity score.

    Args:
    documents (array-like): List of document strings.
    cosine_similarities (numpy.ndarray): Array containing cosine similarity scores.

    Returns:
    list: List of tuples containing resume index and average similarity score, sorted by score in descending order.
    """
    ranked_resumes = []
    for i, resume in enumerate(documents):
        similarity_scores = cosine_similarities[:, i]
        average_similarity_score = similarity_scores.mean()
        ranked_resumes.append((i, average_similarity_score))
    
    ranked_resumes.sort(key=lambda x: x[1], reverse=True)
    return ranked_resumes

def print_top_n_resumes(df, ranked_resumes, n):
    """
    Prints the top N ranked resumes along with their similarity scores.

    Args:
    df (pandas.DataFrame): DataFrame containing resumes.
    ranked_resumes (list): List of tuples containing resume index and average similarity score.
    n (int): Number of top resumes to print.
    """
    if len(ranked_resumes) < n:
        print(f"Error: Number of available resumes ({len(ranked_resumes)}) is less than {n}")
        return
    
    print(f"Top {n} ranked resumes:")
    for i, (index, similarity_score) in enumerate(ranked_resumes[:n], start=1):
        print(f"Rank {i}: Resume Index: {index}, Similarity Score: {similarity_score}")
        print(df.iloc[index]['cleaned_resume'])
        print("-------------------------------------")