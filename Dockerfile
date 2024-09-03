# Use an official Python runtime as a parent image
FROM python:3.9-slim

RUN apt-get update -y
RUN apt-get install -y gcc g++

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the SpaCy large English model
RUN python -m spacy download en_core_web_sm

# Copy the data directory into the container
COPY data/privacy_law_corpus-original_english_text_files /data/corpus

EXPOSE 8501
EXPOSE 8502

# Copy all scripts into the container
COPY scripts/*.py /app/
