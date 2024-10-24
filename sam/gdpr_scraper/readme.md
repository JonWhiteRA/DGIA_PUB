# GDPR Scraper

Scrapes and saves data from https://gdpr-info.eu/

Text files of data currently saved to data/gdpr/

## Files

| File | Purpose |
| ------ | ------ |
| Dockerfile | To build Docker image for scripts |
| file_creation.py | Create text files based on csv file input |
| gdpr_articles.csv | Generated csv file from scraper.py |
| requirements.txt | Python module requirements for Docker image |
| run.sh | Bash script to run Python files |
| scraper.py | Scrape url above and save contents to csv file | 
| visualizer.py | Simple Streamlit app to visualize connections based on direct links |

## To Run

`docker build -t {image_name} .`

`docker run --rm -v {volume_name}:/output {image_name}`

`docker run --rm -v {volume_name}:/output -p 8501:8501 {image_name} streamlit run visualizer.py`