import pandas as pd
import os

def by_paragraph(output_dir, csv_filename):

    # Make output dir if it doesn't exist!
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_filename)

    # Read each row
    for index, row in df.iterrows():
        chapter_num = row['Chapter']
        if int(chapter_num) < 10:
            chapter_num = '0' + str(chapter_num)
        article_num = row['Article Number']
        paragraph_num = str(row['Paragraph Num']).replace('.', '_')
        
        # Make filename
        filename = f"{chapter_num}_{article_num}_{paragraph_num}.txt"
        
        # Specify content
        content = row['Content']
        
        # Save content
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)

def by_article(output_dir, csv_filename):
    # Make output dir if it doesn't exist!
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_filename)

    # Group by article
    grouped = df.groupby('Article Number')

    for article_num, group in grouped:
        # Make the filename
        chapter_num = group['Chapter'].iloc[0]
        if int(chapter_num) < 10:
            chapter_num = '0' + str(chapter_num)
        filename = f"{chapter_num}_{article_num}.txt"
        
        # Start with the title
        content = f"Article {article_num}\n\n"

        # Preserve <li> and <ol> formatting
        for _, row in group.iterrows():
            content += f"Paragraph {row['Paragraph Num']}:\n"
            content += f"• {row['Content']}\n\n"  # Using bullet points for <li>

        # Write content
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content.strip())

if __name__ == '__main__':
    # File to read data from
    csv = '/output/gdpr_articles.csv'
    # Where to save text files (separated by paragraph)
    paragraph_save = '/output/gdpr_by_paragraph/'
    # Where to save text files (separated by article)
    article_save = '/output/gdpr_by_article'

    # Generate text files
    by_article(paragraph_save, csv)
    by_paragraph(article_save, csv)
