import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]',r' ', resume_text)
    resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace
    return resume_text

def extract_features(text_data):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1300)
    word_vectorizer.fit(text_data)
    word_features = word_vectorizer.transform(text_data)
    return word_features

def label_encode_categories(categories):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(categories)
    return encoded_labels