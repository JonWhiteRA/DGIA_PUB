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

## Current Algorithms
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

Build image fresh:
`docker build --no-cache -t {imageName} .`

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

## Metrics for Clustering
1. Silhouette Score: how similar an object is to its own cluster. values range from -1 to +1 where higher values signify better-defined clusters
2. Davies-Bouldin Index: calculates average similarity ratio of each cluster with its most similar where lower values indicate better clustering
3. Dunn Index: ration of minimum inter-cluster distance to maximum intra-cluster distance where high values signify better clustering
4. Inertia/Within-cluster Sum of Squares: measures compactness of clusters with lower values suggesting tighter clusters

## Clustering Algorithms

1. **Agglomerative Clustering**:
   - **Type**: Hierarchical Clustering
   - **Overview**: A bottom-up approach where each data point starts in its own cluster, and pairs of clusters are merged as you move up the hierarchy.
   - **Use Cases**: Customer segmentation, document clustering.
   - **Implemented?:** yes, sam/scripts/agglomerative_clustering.py
   - **Thoughts:** creates many clusters and with few documents in each

2. **Gaussian Mixture Models (GMM)**:
   - **Type**: Probabilistic Clustering
   - **Overview**: Models the data as a mixture of several Gaussian distributions, providing a probabilistic approach to clustering.
   - **Use Cases**: Anomaly detection, image processing.
   - **Implemented?:** yes, sam/scripts/GMM_clustering.py
   - **Thoughts:** able to get exact amount of clusters you want!

3. **Mean Shift**:
   - **Type**: Centroid-based Clustering
   - **Overview**: Identifies clusters by shifting data points towards the densest regions in the feature space.
   - **Use Cases**: Image segmentation, object tracking.
   - **Implemented?:** yes, sam/scripts/mean_shift_clustering.py
   - **Thoughts:** not sure about the Bandwidth slider, larger values mean fewer clusters

4. **Spectral Clustering**:
   - **Type**: Graph-based Clustering
   - **Overview**: Uses the eigenvalues of a similarity matrix to reduce dimensionality before applying a clustering algorithm like K-Means.
   - **Use Cases**: Social network analysis, image segmentation.
   - **Implemented?:** yes, sam/scripts/spectral_clustering.py
   - **Thoughts:** the clusters that it creates don't quite make sense to me :(

### Topic Modeling Algorithms

5. **Non-Negative Matrix Factorization (NMF)**:
   - **Overview**: Factorizes a matrix into two non-negative matrices, often used for topic modeling in text data.
   - **Use Cases**: Document clustering, image processing.

6. **Latent Semantic Analysis (LSA)**:
   - **Overview**: Uses Singular Value Decomposition (SVD) to reduce the dimensionality of term-document matrices, revealing latent topics.
   - **Use Cases**: Information retrieval, recommendation systems.

7. **Hierarchical Dirichlet Process (HDP)**:
   - **Overview**: An extension of LDA that allows for an infinite number of topics, adapting to the complexity of the dataset.
   - **Use Cases**: Topic modeling in large document collections.

### Ensemble Methods

8. **HDBSCAN (Hierarchical DBSCAN)**:
   - **Overview**: An extension of DBSCAN that creates a hierarchy of clusters and is better at handling varying densities.
   - **Use Cases**: Similar use cases as DBSCAN, especially in datasets with varying density.
## Alternate Datasets
All of these sources list pdfs, where a web scraper could be used to download all or hundreds of documents
1. GovInfo - added to project
	- Code Of Federal Regulations collection's Grants and Agreements might be a good option
		- US federal government documents
	- Why?
		- Able to assume the documents have not been translated
		- Possible that grant documents are similar to policy and compliance documents because they outline what must be done for an organization to receive the grant
	- [Link](https://www.govinfo.gov/app/search/%7B%22query%22%3A%22%22%2C%22offset%22%3A0%2C%22pageSize%22%3A100%2C%22historical%22%3Afalse%2C%22facetToExpand%22%3A%22dochierarchy%22%2C%22facets%22%3A%7B%22accodenav%22%3A%5B%22CFR%22%5D%2C%22dochierarchy%22%3A%5B%222%20CFR%20-%20Grants%20and%20Agreements%22%5D%7D%2C%22filterOrder%22%3A%5B%22accodenav%22%2C%22dochierarchy%22%5D%2C%22sortBy%22%3A%220%22%2C%22isLoading%22%3Afalse%7D) 
2. Congress
	- Laws from the 113 to 188 Congress (2013 to present)
	- Why?
		- Documents not translated, all from the federal government
		- Examples of federal policy/law documents
	- Problem! must use API, it blocks scanners: https://www.congress.gov/robots.txt
	- [Link](https://www.congress.gov/advanced-search/legislation?congressGroup%5B0%5D=0&congresses%5B0%5D=118&congresses%5B1%5D=117&congresses%5B2%5D=116&congresses%5B3%5D=115&congresses%5B4%5D=114&congresses%5B5%5D=113&congresses%5B6%5D=112&congresses%5B7%5D=111&congresses%5B8%5D=110&congresses%5B9%5D=109&congresses%5B10%5D=108&congresses%5B11%5D=107&congresses%5B12%5D=106&congresses%5B13%5D=105&congresses%5B14%5D=104&congresses%5B15%5D=103&congresses%5B16%5D=102&congresses%5B17%5D=101&congresses%5B18%5D=100&congresses%5B19%5D=99&congresses%5B20%5D=98&congresses%5B21%5D=97&congresses%5B22%5D=96&congresses%5B23%5D=95&congresses%5B24%5D=94&congresses%5B25%5D=93&legislationNumbers=&restrictionType=field&restrictionFields%5B0%5D=allBillTitles&restrictionFields%5B1%5D=summary&summaryField=billSummary&enterTerms=&wordVariants=true&legislationTypes%5B0%5D=hr&legislationTypes%5B1%5D=hres&legislationTypes%5B2%5D=hjres&legislationTypes%5B3%5D=hconres&legislationTypes%5B4%5D=hamdt&legislationTypes%5B5%5D=s&legislationTypes%5B6%5D=sres&legislationTypes%5B7%5D=sjres&legislationTypes%5B8%5D=sconres&legislationTypes%5B9%5D=samdt&legislationTypes%5B10%5D=suamdt&public=true&private=true&chamber=all&actionTerms=&legislativeActionWordVariants=true&actionDateBoxCount=2&actionText=&dateOfActionOperator=equal&dateOfActionStartDate=&dateOfActionEndDate=&dateOfActionIsOptions=yesterday&actionText2=&dateOfActionOperator2=equal&dateOfActionStartDate2=&dateOfActionEndDate2=&dateOfActionIsOptions2=yesterday&dateOfActionToggle=multi&legislativeAction=151&sponsorState=One&member=&sponsorTypes%5B0%5D=sponsor&sponsorTypes%5B1%5D=sponsor&sponsorTypeBool=OR&dateOfSponsorshipOperator=equal&dateOfSponsorshipStartDate=&dateOfSponsorshipEndDate=&dateOfSponsorshipIsOptions=yesterday&committeeActivity%5B0%5D=0&committeeActivity%5B1%5D=3&committeeActivity%5B2%5D=11&committeeActivity%5B3%5D=12&committeeActivity%5B4%5D=4&committeeActivity%5B5%5D=2&committeeActivity%5B6%5D=5&committeeActivity%5B7%5D=9&satellite=&search=&submitted=Submitted&q=%7B%22bill-status%22%3A%22law%22%2C%22congress%22%3A%5B%22113%22%2C%22114%22%2C%22115%22%2C%22116%22%2C%22117%22%2C%22118%22%5D%7D)
3. Howard County Public Schools - added to project
	- Policies and implementation guides for the local school system
	- Why?
		- Has both the policy and how it is enacted
		- Can help test how we map the changes in one document affecting another
		- Documents not translated but come from the county school board instead of the federal government
	- [Link](https://policy.hcpss.org/)