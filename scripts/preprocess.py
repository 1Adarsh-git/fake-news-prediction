import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(
            token.lower() for token in str(sentence).split()
            if token not in stopwords.words('english')
        ))
    return preprocessed_text
