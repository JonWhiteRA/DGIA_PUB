# Use an official Python runtime as a parent image
FROM python:3.9-slim-bullseye

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y
RUN apt-get install -y gcc g++

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download the SpaCy large English model
RUN python -m spacy download en_core_web_sm

# Download specific Hugging Face models to the Docker image
RUN python -c "from sentence_transformers import SentenceTransformer; \
                SentenceTransformer('all-MiniLM-L6-v2');"

# Download necessary NLTK datasets
RUN python -c "import nltk; \
               nltk.download('stopwords'); \
               nltk.download('punkt'); \
               nltk.download('punkt_tab'); \
               nltk.download('words')"

# Copy the data directory into the container
# COPY data/privacy_law_corpus-original_english_text_files /data/corpus

# Copy new data into container
COPY data/govInfo /data/gov
COPY data/hcpss /data/hcpss
COPY data/gdpr /data/corpus

# COPY scripts/output /data/output

ENV CORPUS_PATH="/data/corpus"
ENV OUTPUT_PATH="/app/output"

EXPOSE 8501
EXPOSE 8502

# Copy all scripts into the container
COPY scripts/*.py /app/

# Copy scripts from sam folder
COPY sam/scripts/*_clustering.py /app/
COPY sam/scripts/app.py /app/
COPY sam/scripts/metrics.py /app/
COPY sam/scripts/tools.py /app/
COPY sam/scripts/multi* /app/

# Save output to volume
VOLUME /app/output