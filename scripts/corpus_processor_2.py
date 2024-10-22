import os
import json
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from read_data_files import read_files_in_directory

# Function to find overlaps between sets of data (keywords or entities)
def find_overlaps(data):
    overlaps = {}
    keys = list(data.keys())
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1 = keys[i]
            key2 = keys[j]
            set1 = set(data[key1])
            set2 = set(data[key2])
            overlap_count = len(set1 & set2)
            total_unique = len(set1 | set2)
            crude_score = overlap_count / total_unique if total_unique != 0 else 0
            overlaps[(key1, key2)] = {
                "overlap_count": overlap_count,
                "total_unique": total_unique,
                "crude_score": crude_score
            }
            
    return overlaps

# Function to sort overlaps by count
def sort_overlaps_by_count(overlaps):
    sorted_overlaps = sorted(overlaps.items(), key=lambda item: item[1]['overlap_count'], reverse=True)
    return sorted_overlaps

# Function to get top related files based on crude score
def get_top_related_files(overlaps, top_n=10):
    related_files = {}
    for (file1, file2), data in overlaps.items():
        if file1 not in related_files:
            related_files[file1] = []
        if file2 not in related_files:
            related_files[file2] = []
        related_files[file1].append((file2, data['crude_score']))
        related_files[file2].append((file1, data['crude_score']))
    
    # Sort each list of related files by crude score in descending order and take top N
    for file in related_files:
        related_files[file].sort(key=lambda x: x[1], reverse=True)
        related_files[file] = related_files[file][:top_n]
    
    return related_files

# Function to generate embeddings for the documents
def generate_embeddings(data_files_dict, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings_dict = {}

    # Iterate over all files in the directory
    for filename in data_files_dict :
        text = data_files_dict[filename]

        # Replace newlines with spaces in the entire text
        text = text.replace('\n', ' ').replace('\r', ' ')

        # Generate the embedding for the document
        embedding = model.encode(text)

        # Store the embedding in the dictionary with the filename as the key
        embeddings_dict[filename] = embedding.tolist()

    return embeddings_dict

# Function to generate top related embeddings file based on cosine similarity
def generate_top_related_embeddings(embeddings_dict, top_n=10, output_dir="."):
    similarity_dict = {}

    # Get a list of filenames for easy indexing
    filenames = list(embeddings_dict.keys())

    # Convert the embeddings_dict to a numpy array for easier computation
    embedding_matrix = np.array([embeddings_dict[filename] for filename in filenames])

    # Compute the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(embedding_matrix)

    # Populate the similarity_dict with top N similar documents for each file
    for i, filename in enumerate(filenames):
        # Get the cosine similarities for the current document
        cosine_similarities = cosine_sim_matrix[i]
        
        # Get the top N similar documents (excluding the document itself)
        similar_indices = cosine_similarities.argsort()[-(top_n + 1):-1][::-1]  # Top N, excluding the file itself
        
        # Create a list of lists [filename, cosine_similarity] for the top N
        top_similar = [[filenames[j], float(cosine_similarities[j])] for j in similar_indices]
        
        # Add to the similarity_dict
        similarity_dict[filename] = top_similar

    # Convert the similarity_dict to JSON and save to a file
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, "top_related_files_similarity.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(similarity_dict, f, ensure_ascii=False, indent=4)

    print(f"Document similarity JSON with top {top_n} similar documents saved successfully to {json_output_path}.")

def make_absolute_path(directory, filename):
    # Join the directory and filename to create a path
    path = os.path.join(directory, filename)
    # Convert the path to an absolute path
    absolute_path = os.path.abspath(path)
    return absolute_path

# Main function to load keywords and entities, compute overlaps, and generate top related embeddings
def main(keywords_file, entities_file, embeddings_file, directory, output_dir, top_n=10):
    # Load JSON data for keywords and entities
    with open(keywords_file, 'r') as f:
        keywords = json.load(f)

    with open(entities_file, 'r') as f:
        entities = json.load(f)

    # Generate or load embeddings
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'r') as f:
            embeddings_dict = json.load(f)
        print(f"Loaded embeddings from {embeddings_file}.")
    else:
        data_files_dict = read_files_in_directory(directory)

        embeddings_dict = generate_embeddings(data_files_dict)
        embeddings_file = 'output_embeddings.json'
        os.makedirs(output_dir, exist_ok=True)
        embeddings_file_path = os.path.join(output_dir, embeddings_file)
        with open(embeddings_file_path, 'w') as f:
            json.dump(embeddings_dict, f, indent=4)
        print(f"Embeddings have been saved to {embeddings_file_path}.")

    # Prepare dictionaries for keywords and entities
    fk_dict = {file: [t[0] for t in keywords[file]] for file in keywords}
    fe_dict = {file: [t[0] for t in entities[file]] for file in entities}

    # Compute overlaps for keywords and entities
    keyword_overlaps = find_overlaps(fk_dict)
    entity_overlaps = find_overlaps(fe_dict)

    # Sort overlaps
    sorted_keyword_overlaps = sort_overlaps_by_count(keyword_overlaps)
    sorted_entity_overlaps = sort_overlaps_by_count(entity_overlaps)

    # Convert sorted overlaps to the desired output format
    output_list_keywords = [
        {
            "file1": pair[0][0],
            "file2": pair[0][1],
            "overlap_count": pair[1]['overlap_count'],
            "total_unique": pair[1]['total_unique'],
            "crude_score": pair[1]['crude_score']
        }
        for pair in sorted_keyword_overlaps
    ]

    output_list_entities = [
        {
            "file1": pair[0][0],
            "file2": pair[0][1],
            "overlap_count": pair[1]['overlap_count'],
            "total_unique": pair[1]['total_unique'],
            "crude_score": pair[1]['crude_score']
        }
        for pair in sorted_entity_overlaps
    ]

    # Generate the top related files dictionary
    top_related_files_keywords = get_top_related_files(keyword_overlaps, top_n=top_n)
    top_related_files_entities = get_top_related_files(entity_overlaps, top_n=top_n)

    # Write the output files for keywords
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'scored_keywords.json'), 'w') as file:
        json.dump(output_list_keywords, file, indent=4)

    with open(os.path.join(output_dir, 'top_related_files_keywords.json'), 'w') as file:
        json.dump(top_related_files_keywords, file, indent=4)

    # Write the output files for entities
    with open(os.path.join(output_dir, 'scored_entities.json'), 'w') as file:
        json.dump(output_list_entities, file, indent=4)

    with open(os.path.join(output_dir, 'top_related_files_entities.json'), 'w') as file:
        json.dump(top_related_files_entities, file, indent=4)

    # Generate top related embeddings file based on cosine similarity
    generate_top_related_embeddings(embeddings_dict, top_n=top_n, output_dir=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute overlaps between keywords and entities from JSON files, generate embeddings if they don't exist, and generate top related embeddings file.")
    parser.add_argument("--input_dir", type=str, help="The directory containing the text files for embedding generation.")
    parser.add_argument("--top_n", type=int, default=10, help="The number of top related files to retrieve. Default is 10.")
    parser.add_argument("--output_dir", type=str, default="output", help="The output directory for saving results. Default is 'output'.")

    args = parser.parse_args()

    keywords_file = make_absolute_path(args.output_dir, "keywords.json")
    entities_file = make_absolute_path(args.output_dir, "entities.json")
    embeddings_file = make_absolute_path(args.output_dir, "embeddings.json")

    main(keywords_file, entities_file, embeddings_file, args.input_dir, args.output_dir, top_n=args.top_n)
