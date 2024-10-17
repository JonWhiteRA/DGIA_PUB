## Goal
identify and resolve conflicts in policy documents to assess impacts of new government policies

## Idea
use unclassified legal documents as test in order to find methods to create and analyze semantic relationships

## Academic Research
- Applying [BERT](notes/BERT.md) Embeddings to Predict Legal Textual Entailment: [document](notes/s12626-022-00101-3.pdf) notes
- PolicyGPT: Automated Analysis of Privacy Policies with Large Language Models: [document](notes/2309.10238v1.pdf) notes
- Creation and Analysis of an International Corpus of Privacy Laws: [document](notes/2024.lrec-main.365.pdf) notes
- Similar Cases Recommendation using Legal Knowledge Graphs: [document](notes/2107.04771v2.pdf) notes

## Approach
1) Find documents related to a topic - not a keyword but the topic! Potential technique: Document Level Topic Modelling and Indexing
2) Given a new policy document, find impacted compliance and standards documents. Potential technique: Logical Document Semantic Comparison
3) Find documents whose changes would **most** impact other documents, standards, etc. along with identifying nature of said changes. Potential technique: Fine Tuning across multiple document types
4) Identify conflicting requirements or policy documents. Potential technique: Fine Tuning and Data Comparisons
5) Find substantial and insubstantial changes in documents. Potential technique: In-Document Semantic Similarity and Relational Weighting
6) Find requirement documents that would impact a new system development effort. Potential technique: Fine Tuning
7) Find documents that have low impact on others, but combined could have most influence. Potential technique: Semantic Understanding

## Algorithms
1) [Latent Dirichlet Allocation](<notes/Latent Dirichlet Allocation>) (LDA) - unsupervised clustering of text that attempts to group words and documents into a predefined number of clusters. these clusters represent individual topics.
2) [Density-Based Spatial Clustering of Applications with Noise](<notes/Density-Based Spatial Clustering of Applications with Noise.md>) (DBSCAN) - finds core samples of high density and expands clusters from them.
3) [K-means](notes/K-means) - group similar text data like documents, sentences, words, etc. into clusters

## Python Modules
- [NetworkX](https://networkx.org/documentation/stable/index.html) (graph_file_analysis_app) - 
	- [girvan-newman](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html) - finds communities in a graph using the Girvan-Newman method
	- [louvain_communities](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html#networkx.algorithms.community.louvain.louvain_communities) - find best partition of a graph using Louvain Community Detection Algorithm
- [tqdm]([tqdm · PyPI](https://pypi.org/project/tqdm/)) (corpus_processor_1) - progress  bar
- [nltk](https://www.nltk.org/) (corpus_processor_1) - Natural Language Toolkit
	- interfaces for many corpora and lexical resources
	- text processing libraries
	- Libraries used:
		- Corpus - readers for groups of written text
		- Tokenize - split text into words or phrases
- [sklearn](https://scikit-learn.org/stable/) (corpus_processor_1, corpus_processor_2, lda_file_analysis_app) - predictive data analysis
	- machine learning for supervised and unsupervised learning
	- model fitting, data preprocessing, model selection and evaluation, etc.
	- preprocessing includes converting terms to root lexical forms (ex: barks, barking, barked --> bark)
	- stop words are a list of words taken out of documents beforehand (ex: the, that, are, is, etc.)
	- libraries used:
		- CountVectorizer - text feature extraction
		- TfidfTransformer - transform count matrix into normalized term-frequency (tf) or term-frequency inverse document frequency (tf-idf)
			- tf = how many times a word appears in a document
			- tf-idf = how rare a word is across collection of documents
		- LatentDirichletAllocation - "Latent Dirichlet Allocation with online variational Bayes algorithm"
			- [Latent Dirichlet Allocation](<notes/Latent Dirichlet Allocation>): generative statistical model for automatically extracting topics in textual corpora
			- online variational Bayes algorithm - estimates posterior of complex hierarchical Bayesian models
				- online = problems have no or incomplete knowledge of the future (ex: the ski problem, do you buy skis for a one-time cost or rent them at a lower but repeated cost? you'd have to guess how many times you go skiing and then if it would be cheaper to rent or buy the skis. this is online because you do not know how many times you will go skiing so your information is incomplete)
				- offline = complete information assumed
		- TSNE - [T-distributed Stochastic Neighbor Embedding (t-SNE)](notes/t-SNE.md)
			- way to reduce dimensions so you can visualize high-dimensionality data in two or three dimensions
			- used to reveal clusters and patterns in complex datasets
		- cosine_similarity - computes the normalized dot product of X and Y
- [spacy](https://spacy.io/usage/spacy-101/) (corpus_processor_1) - natural language processing
	- supports tokenization and training for 70+ languages
	- neural network models for tagging, parsing, named entity recognition, text classification, etc.
	- multi-task learning with [BERT](notes/BERT.md)
- [pickle](https://docs.python.org/3/library/pickle.html) (corpus_processor_1, lda_file_analysis_app) - serializes and de-serializes Python object structures
- [pandas](https://pandas.pydata.org/) (corpus_processor_1, graph_file_analysis, lda_file_analysis_app) - data structures for relational or labelled data usage
- [numpy](https://numpy.org/learn/) (corpus_processor_2) - support for large, multi-dimensional arrays and matrices
- [sentence_transformers](https://sbert.net/) (corpus_processor_2) - transform sentences and paragraphs into vector representations (embeddings) using pre-trained models based on [BERT](notes/BERT.md)
- [plotly](https://plotly.com/python/) (graph_file_analysis_app, lda_file_analysis_app) - graphing
- [streamlit](https://streamlit.io/) (graph_file_analysis_app, lda_file_analysis_app) - easy way to deploy ML model and Python without worrying about frontend
- [scipy](https://docs.scipy.org/doc/scipy/index.html) (lda_file_analysis_app) - scientific and technical computing
	- libraries used:
		- distance - distance matrix calculated from collection of observation vectors stored in rectangular array
- [docx2txt](https://pypi.org/project/docx2txt/) (read_files) - extract text from docx
- [textract](https://textract.readthedocs.io/en/stable/) (read_files) - extract text from various file types
- [PyPDF2](https://pypi.org/project/PyPDF2/) (read_files) - extract text from PDFs
- [argparse](https://docs.python.org/3/library/argparse.html) (read_files) - parse command-line options, arguments, and subcommands
- [json](https://docs.python.org/3/library/json.html) (read_files) - JSON encoder and decoder
- [re](https://docs.python.org/3/library/re.html) (read_files) - regular expression matching operations

## GitHub Files
**/scripts/corpus_processor_1.py:** process directory of text files, extracting keywords and named entities. Performs topic modelling with LDA and t-SNE dimensionality reduction. 

**/scripts/corpus_processor_2.py:** process directory of text files to compute keyword and entity overlaps between them. Generates embeddings using pre-trained model, then finds and ranks the most related files based on embeddings.

## Running Docker

Create volume:
`docker volume create {volumeName}`

Build image:
`docker build -t {imageName} .`

Run corpus_processor_1.py with volume attached: 
`docker run -v {volumeName}:/app/output -it {imageName} python3 corpus_processor_1.py /data/corpus`

Run corpus_processor_2.py using output from corpus_processor_1.py:
`docker run -v {volumeName}:/app/output -it {imageName} python3 corpus_processor_2.py /data/corpus`

Run graph_file_analysis.py:
`docker run -v {volumeName}:/app/output -it {imageName} python3 graph_file_analysis_app.py`

Locally host webpage to view graphs:
`docker run -v {volumeName}:/app/output -p 8501:8501 -it {imageName} streamlit run graph_file_analysis_app.py`

Go to `http://localhost:8501` to view graphs!

:)