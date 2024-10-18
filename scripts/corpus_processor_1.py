import os
import sys
import argparse
from tqdm import tqdm
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
import spacy
import json
from collections import defaultdict
import pickle
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import PyPDF2

# Download the necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('words')

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Increase spaCy's max length limit
nlp.max_length = 2000000  # Adjust this value as needed

# Function to read text from a file
def read_text(file_path):
    # Check the file extension
    if file_path.lower().endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        return read_txt(file_path)
    else:
        raise ValueError("Unsupported file type: {}".format(file_path))

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page_text = reader.pages[page_num].extract_text() or ""
                text += page_text + "\n"  # Add a newline for each page
            return normalize_text(text)
    except Exception as e:
        return f"Error reading PDF file: {e}"

def normalize_text(text):
    # Normalize whitespace and return
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

# Function to extract top N keywords with their counts
def extract_keywords(files_dict, top_n=25):
    stop_words = set(stopwords.words('english'))
    valid_words = set(words.words())
    keywords_dict = {}

    # Adding progress bar for keyword extraction
    for filename, text in tqdm(files_dict.items(), desc="Extracting keywords", unit="file"):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words and word in valid_words]

        if not tokens:  # If no tokens left after filtering
            print(f"Warning: {filename} resulted in an empty token list. Skipping...")
            keywords_dict[filename] = []
            continue

        try:
            cv = CountVectorizer(max_df=1.0, max_features=10000)
            word_count_vector = cv.fit_transform([' '.join(tokens)])

            if len(cv.get_feature_names_out()) == 0:
                print(f"Warning: {filename} resulted in an empty feature set after vectorization. Skipping...")
                keywords_dict[filename] = []
                continue

            tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
            tfidf_transformer.fit(word_count_vector)

            feature_names = cv.get_feature_names_out()
            tfidf_vector = tfidf_transformer.transform(word_count_vector)
            sorted_items = sorted(zip(tfidf_vector.toarray()[0], feature_names), reverse=True)

            top_keywords = sorted_items[:top_n]
            
            # Calculate keyword counts
            keyword_counts = cv.transform([' '.join(tokens)]).toarray().flatten()
            keywords_with_counts = [(keyword, int(keyword_counts[idx])) for score, keyword in top_keywords for idx in range(len(feature_names)) if feature_names[idx] == keyword]
            
            # Sort keywords by count in descending order
            keywords_with_counts = sorted(keywords_with_counts, key=lambda x: x[1], reverse=True)
            
            keywords_dict[filename] = keywords_with_counts

        except ValueError as e:
            print(f"Error processing {filename}: {e}")
            keywords_dict[filename] = []
            continue

    return keywords_dict

# Function to extract entities from text using spaCy
def extract_entities(files_dict):
    entities_dict = {}

    # Adding progress bar for entity extraction
    for filename, text in tqdm(files_dict.items(), desc="Extracting entities", unit="file"):
        try:
            doc = nlp(text.replace('\n', ' ').replace('\r', ' '))
            entity_count = defaultdict(int)
            for ent in doc.ents:
                if ent.label_ != "CARDINAL":  # Skip CARDINAL entities
                    entity_count[(ent.text.replace('\n', ' ').replace('\r', ' '), ent.label_)] += 1
            
            # Convert entity counts to a list of tuples with counts
            deduped_entities = [(text, label, count) for (text, label), count in entity_count.items()]
            
            # Sort the list of deduplicated entities with counts in descending order by count
            deduped_entities.sort(key=lambda x: x[2], reverse=True)
            
            entities_dict[filename] = deduped_entities
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            entities_dict[filename] = []
            continue

    return entities_dict

# Function to load or calculate LDA and t-SNE results
def load_or_calculate_lda_tsne(files_dict, num_topics=5, n_top_words=10, output_dir="."):
    documents = list(files_dict.values())
    filenames = list(files_dict.keys())
    
    tsne_file = os.path.join(output_dir, f"tsne_results_topics_{num_topics}_words_{n_top_words}.pkl")
    
    if os.path.exists(tsne_file):
        with open(tsne_file, 'rb') as f:
            df_tsne = pickle.load(f)
        print(f"Loaded t-SNE results from {tsne_file}")
    else:
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(documents)

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        lda.fit(X)

        tf_feature_names = vectorizer.get_feature_names_out()
        topic_keywords = print_top_words(lda, tf_feature_names, n_top_words)

        doc_topic_distr = lda.transform(X)
        df_tsne = pd.DataFrame(doc_topic_distr, columns=[f'Topic {i}' for i in range(num_topics)])
        df_tsne['dominant_topic'] = df_tsne.idxmax(axis=1)

        tsne_model = TSNE(n_components=2, random_state=0, perplexity=15, n_iter=300)
        tsne_values = tsne_model.fit_transform(doc_topic_distr)

        df_tsne['x'] = tsne_values[:, 0]
        df_tsne['y'] = tsne_values[:, 1]
        df_tsne['filename'] = filenames
        df_tsne['topic_keywords'] = df_tsne['dominant_topic'].map(lambda x: topic_keywords[int(x.split(' ')[1])])

        with open(tsne_file, 'wb') as f:
            pickle.dump(df_tsne, f)
        print(f"Saved t-SNE results to {tsne_file}")
    
    return df_tsne

# Helper function to print the top words per topic
def print_top_words(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(" ".join(top_words))
    return topics

def process_directory(directory):
    # Dictionary to hold the filename and text content
    files_dict = {}

    # Get a list of all .txt files in the directory
    txt_files = [filename for filename in os.listdir(directory) if filename.endswith('.txt')]
    txt_files = txt_files + [filename for filename in os.listdir(directory) if filename.endswith('.pdf')]

    # Loop through each file in the directory and read the content with a progress bar
    for filename in tqdm(txt_files, desc="Reading files", unit="file"):
        file_path = os.path.join(directory, filename)
        text = read_text(file_path)
        files_dict[filename] = text

    return files_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of text files, extract keywords, entities, and LDA topics.")
    parser.add_argument("directory", type=str, help="The path to the directory containing the text files.")
    parser.add_argument("--top_n", type=int, default=25, help="The number of top keywords to extract. Default is 25.")
    parser.add_argument("--num_topics", type=int, default=5, help="The number of topics for LDA. Default is 5.")
    parser.add_argument("--n_top_words", type=int, default=10, help="The number of top words per topic. Default is 10.")
    parser.add_argument("--output_dir", type=str, default="output", help="The output directory for saving results. Default is 'output'.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    files_dict = process_directory(args.directory)
    
    # Check if keywords.json exists, if not, extract keywords
    keywords_file = os.path.join(args.output_dir, "keywords.json")
    if os.path.exists(keywords_file):
        with open(keywords_file, 'r') as f:
            keywords_dict = json.load(f)
        print(f"Loaded keywords from {keywords_file}")
    else:
        keywords_dict = extract_keywords(files_dict, top_n=args.top_n)
        with open(keywords_file, 'w') as f:
            json.dump(keywords_dict, f, indent=2)
        print(f"Keywords have been saved to {keywords_file}")
    
    # Check if entities.json exists, if not, extract entities
    entities_file = os.path.join(args.output_dir, "entities.json")
    if os.path.exists(entities_file):
        with open(entities_file, 'r') as f:
            entities_dict = json.load(f)
        print(f"Loaded entities from {entities_file}")
    else:
        entities_dict = extract_entities(files_dict)
        with open(entities_file, 'w') as f:
            json.dump(entities_dict, f, indent=2)
        print(f"Entities have been saved to {entities_file}")
    
    # Calculate or load LDA and t-SNE results
    lda_tsne_results = load_or_calculate_lda_tsne(files_dict, num_topics=args.num_topics, n_top_words=args.n_top_words, output_dir=args.output_dir)
