# rankings
 
ranker.py contains all the main functionality of the ranking system.
rankings.ipynb runs a complete pipeline and prints the top n resumes.
Set n accordingly.

This code constitutes a pipeline for processing resume data, computing cosine similarity between resumes and a set of keywords, and ranking the resumes based on their similarity scores. Initially, the preprocess_data function reads resume data from a CSV file, removes duplicate entries, resets the DataFrame index, and cleans each resume text. Subsequently, the compute_similarity function calculates cosine similarity scores between the cleaned resumes and the provided keywords using TF-IDF vectorization. The sort_resumes_by_similarity function sorts the resumes based on their average cosine similarity scores and returns a list of tuples containing the resume index and its score. Lastly, the print_top_n_resumes function prints the top-ranked resumes along with their similarity scores. 