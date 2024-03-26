import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

def clean_resume(resume_text):
    """
    Cleans the text of a resume by removing URLs, RT and cc tags, hashtags, mentions, punctuations, 
    non-ASCII characters, and extra whitespace.

    Args:
    resume_text (str): The raw text of a resume.

    Returns:
    str: Cleaned text of the resume.
    """
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]',r' ', resume_text)
    resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace
    return resume_text

def extract_features(text_data):
    """
    Extracts features from text data using TF-IDF vectorization.

    Args:
    text_data (list): List of text strings.

    Returns:
    scipy.sparse.csr_matrix: TF-IDF matrix of word features.
    """
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1300)
    word_vectorizer.fit(text_data)
    word_features = word_vectorizer.transform(text_data)
    return word_features

def label_encode_categories(categories):
    """
    Encodes categorical labels using LabelEncoder.

    Args:
    categories (list): List of categorical labels.

    Returns:
    numpy.ndarray: Encoded labels.
    """
    le = LabelEncoder()
    encoded_labels = le.fit_transform(categories)
    return encoded_labels