import sys
import os
import re
import xml.etree.ElementTree as ET
import mwparserfromhell
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from spellchecker import SpellChecker

print ("Downloading nltk punkt... This should take a few minutes...")
nltk.download('punkt')
print ("Done downloading nltk punkt.")

def parse_args():
    if len(sys.argv) != 3:
        print("Usage: python script.py <xml_dump_file_path> <export_directory>")
        sys.exit(1)

    xml_dump_file_path = sys.argv[1]
    export_directory = sys.argv[2]

    if not os.path.isfile(xml_dump_file_path):
        print("Invalid XML dump file path.")
        sys.exit(1)

    if not os.path.isdir(export_directory):
        print("Invalid export directory.")
        sys.exit(1)

    return xml_dump_file_path, export_directory


def remove_tags_and_non_english_characters(text):
    wikicode = mwparserfromhell.parse(text)
    spell = SpellChecker()

    # Remove citations and other metadata
    for template in wikicode.filter_templates():
        if template.name.strip().lower() in ('ref', 'reflist', 'note', 'citation'):
            try:
                wikicode.remove(template)
            except ValueError:
                pass

    cleaned_text = wikicode.strip_code().strip()
    cleaned_text = re.sub('[^a-zA-Z0-9 \n\.,:;\-\'\"\(\)\[\]\{\}\?\!]+', ' ', cleaned_text)

    # Filter out text that doesn't form complete sentences
    sentences = nltk.sent_tokenize(cleaned_text)
    complete_sentences = [sentence for sentence in sentences if re.match(r'^[A-Z].*[.!?]$', sentence, flags=re.IGNORECASE)]

    # Remove sentences containing non-English words enclosed in parentheses
    filtered_sentences = []
    for sentence in complete_sentences:
        parenthesized_words = re.findall(r'\((.*?)\)', sentence)
        non_english_words = [word for word in parenthesized_words if not re.match(r'^[a-zA-Z0-9 \-\'\"\(\)\[\]\{\}\?\!]+$', word)]
        if not non_english_words:
            filtered_sentences.append(sentence)

    # Remove sentences with more than 50% misspelled words
    checked_sentences = []
    for sentence in filtered_sentences:
        words = sentence.split()
        misspelled = spell.unknown(words)
        if len(misspelled) / len(words) <= 0.5:
            checked_sentences.append(sentence)

    # Remove formatting issues
    cleaned_text = ' '.join(checked_sentences)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)  # Remove multiple spaces
    cleaned_text = re.sub(r'\s*([.,:;!?])\s*', r'\1 ', cleaned_text)  # Remove spaces before and after punctuation marks

    # Convert to lowercase
    cleaned_text = cleaned_text.lower()

    # Remove emails and URLs
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)

    # Remove sentences with punctuation separated by spaces without any English in-between
    cleaned_text = re.sub(r'([.,:;!?\-])\s+([.,:;!?\-])', ' ', cleaned_text)

    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


def find_keywords_in_text(text, keywords, min_keyword_frequency=5, min_tfidf_score=0.2):
    # Check if the keyword appears in the title or summary (first few sentences)
    title_and_summary = ' '.join(nltk.sent_tokenize(text)[:3])
    for keyword in keywords:
        if keyword.lower() in title_and_summary.lower():
            return True

    # Check the keyword frequency in the text
    text_lower = text.lower()
    keyword_frequencies = {keyword: text_lower.count(keyword.lower()) for keyword in keywords}
    frequent_keywords = [keyword for keyword, freq in keyword_frequencies.items() if freq >= min_keyword_frequency]
    if frequent_keywords:
        return True

    # Calculate the TF-IDF score for keywords
    vectorizer = TfidfVectorizer(vocabulary=keywords, lowercase=True, stop_words='english')
    tfidf_scores = vectorizer.fit_transform([text])
    keyword_tfidf_scores = {keyword: tfidf_scores[0, vectorizer.vocabulary_[keyword]] for keyword in keywords}

    # Check if any keyword has a TF-IDF score above the minimum threshold
    high_tfidf_keywords = [keyword for keyword, score in keyword_tfidf_scores.items() if score >= min_tfidf_score]
    if high_tfidf_keywords:
        return True

    return False


def parse_and_export_articles(xml_dump_file_path, export_directory, keywords):
    context = ET.iterparse(xml_dump_file_path, events=("start", "end"))
    context = iter(context)
    event, root = next(context)

    title = None

    print("Beginning to parse XML dump file...")

    loop_iter = 0
    for event, elem in context:
        loop_iter += 1
        if loop_iter % 100000 == 0:
            print(f"Processed {loop_iter} articles.")
            print("Excerpt from last article:" + text[:100])

        if event == "end" and elem.tag.endswith("title"):
            title = elem.text
        elif event == "end" and elem.tag.endswith("text"):
            text = elem.text
            if text is not None and title is not None and find_keywords_in_text(text, keywords):

                print("Exporting article:", title)
                cleaned_text = remove_tags_and_non_english_characters(text)
                safe_article_title = re.sub(r'[\\/:*?"<>|]', '_', title)
                file_path = os.path.join(export_directory, f"{safe_article_title}.txt")

                if len(cleaned_text) < 300:
                    print("Article is too short, skipping.")
                    continue

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)

            root.clear()


computer_science_terms = [
    'algorithms',
    'artificial intelligence',
    'big data',
    'cloud computing',
    'coding theory',
    'compilers',
    'computability theory',
    'computational biology',
    'computational complexity theory',
    'computational geometry',
    'computational linguistics',
    'computer architecture',
    'computer graphics',
    'computer networking',
    'computer security',
    'computer vision',
    'cybersecurity',
    'data science',
    'databases',
    'distributed computing',
    'human-computer interaction',
    'information retrieval',
    'information science',
    'information theory',
    'machine learning',
    'natural language processing',
    'numerical analysis',
    'operating systems',
    'programming languages',
    'quantum computing',
    'software engineering',
    'systems biology',
    'computer science',
    'theoretical computer science'
]

def main():
    xml_dump_file_path, export_directory = parse_args()
    parse_and_export_articles(xml_dump_file_path, export_directory, computer_science_terms)

if __name__ == "__main__":
    main()
