import os
import docx2txt
import textract
import PyPDF2
import argparse
import json
import re

def read_files_in_directory(directory_path):
    files_content = {}
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            print(f"Reading file: {filename}")
            try:
                if filename.endswith(".txt"):
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                        content = file.read()
                        files_content[filename] = strip_non_ascii(content)
                elif filename.endswith(".docx"):
                    content = docx2txt.process(file_path)
                    files_content[filename] = strip_non_ascii(content)
                elif filename.endswith(".pdf"):
                    content = read_pdf(file_path)
                    files_content[filename] = strip_non_ascii(content)
                elif filename.endswith(".doc"):
                    content = textract.process(file_path).decode("utf-8", errors='replace')
                    files_content[filename] = strip_non_ascii(content)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return files_content

def strip_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text() or ""
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read all files in a specified directory.")
    parser.add_argument("--input_dir", default="/data/corpus", help="The directory path to read files from (default: /data/corpus)")
    parser.add_argument("--output_dir", default="/data/output", help="The output directory path for the JSON result (default: /data/output)")
    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir
    output_path = os.path.join(output_directory, "corpus.json")
    documents_content = read_files_in_directory(input_directory)

    # Write the resulting dictionary to the specified output file
    os.makedirs(output_directory, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(documents_content, json_file, ensure_ascii=False, indent=4)

    # Example: Print the contents of each document
    for filename, content in documents_content.items():
        print(f"Filename: {filename}\nContent: {content[:100]}...\n")