import re
import numpy as np
from sklearn.cluster import DBSCAN


def extract_keyword_positions(text, keywords):
    """ Returns a list of tuples (position, keyword) for each keyword found in the text.

    Example Usage:
        >>> extract_keyword_positions("I like apples and oranges", ["apples", "oranges"])
        [(7, "apples"), (18, "oranges")]
        >>> extract_keyword_positions("I like apples and oranges", ["I",{"sequence": ["apples", "oranges"], "max_distance": 100}])
        [(0, 'I'), (7, 'apples'), (18, 'oranges')]

    Args:
        text (str): The text to search.
        keywords (list): A list of keywords to search for. A keyword can be a string or a dict. If a dict, it must have a "sequence" key and an optional "max_distance" key.
        The "sequence" key must have a list of strings as its value. The "max_distance" key must have an integer as value, specifying the maximum character distance between the keywords in the sequence.

    Returns:
        list: A list of tuples (position, keyword) for each keyword found in the text.
    """
    positions = []
    for keyword in keywords:
        if isinstance(keyword, str):
            for match in re.finditer(re.escape(keyword), text):
                positions.append((match.start(), keyword))
        elif isinstance(keyword, dict) and "sequence" in keyword:
            sequence, max_distance = keyword["sequence"], keyword["max_distance"]
            pattern = r"\b(?:{})\b".format("|".join(re.escape(kw)
                                                    for kw in sequence))  # Matches any of the keywords in the sequence
            matches = list(re.finditer(pattern, text))
            for i in range(len(matches) - len(sequence) + 1):
                seq_matches = matches[i:i + len(sequence)]
                if all(seq_matches[j].group() == sequence[j] for j in range(len(sequence))):
                    if seq_matches[-1].start() - seq_matches[0].start() <= max_distance:
                        # Uses extend instead of append to add multiple items at once, faster than += for lists
                        positions.extend([(match.start(), sequence[j])
                                         for j, match in enumerate(seq_matches)])

    return sorted(positions, key=lambda x: x[0])


def cluster_positions(positions, max_distance):
    """ Clusters positions using DBSCAN. DBSCAN is a density-based clustering algorithm that groups together points that are close to each other.
    It is a good choice for clustering positions because it can handle outliers and does not require the number of clusters to be specified.

    Example Usage:
    >>> cluster_positions([(0, 'I'), (7, 'apples'), (18, 'oranges')], 50)
    [0, 0, 0] # All three points are in the same cluster

    Args:
        positions (list): A list of tuples (position, keyword) for each keyword found in the text.
        max_distance (int): The maximum distance between two positions to be considered in the same cluster.

    Returns:
        list: A list of cluster labels for each position.
    """
    X = np.array([[pos[0]] for pos in positions])
    dbscan = DBSCAN(eps=max_distance, min_samples=2).fit(X)
    return dbscan.labels_


def extract_chunks(text, positions, clusters):
    """ Extracts chunks from the text based on the positions and clusters.

    Example Usage:
    >>> extract_chunks("I like apples and oranges", [(0, 'I'), (7, 'apples'), (18, 'oranges')], [0, 0, 0])
    ['I like apples and oranges']

    Args:
        text (str): The text to search.
        positions (list): A list of tuples (position, keyword) for each keyword found in the text.
        clusters (list): A list of cluster labels for each position.

    Returns:
        list: A list of chunks.
    """
    chunks = []
    # +1 because max(clusters) returns the highest cluster label
    for cluster in range(max(clusters) + 1):
        cluster_positions = [pos for i, pos in enumerate(
            positions) if clusters[i] == cluster]
        if cluster_positions:
            start = cluster_positions[0][0]
            end = cluster_positions[-1][0] + len(cluster_positions[-1][1])
            chunks.append(text[start:end])
    return chunks


def count_keywords(chunk, keywords):
    """ Counts the number of keywords in a chunk.

    Example Usage:
    >>> count_keywords("I like apples and oranges", ["apples", "oranges"])
    2

    Args:
        chunk (str): The chunk of text to search.
        keywords (list): A list of keywords to search for. A keyword can be a string or a dict. If a dict, it must have a "sequence" key and an optional "max_distance" key.

    Returns:
        int: The number of keywords in the chunk.
    """

    count = 0
    for keyword in keywords:
        if isinstance(keyword, str):
            count += len(re.findall(re.escape(keyword), chunk))
        elif isinstance(keyword, dict) and "sequence" in keyword:
            sequence = keyword["sequence"]
            pattern = r"\b(?:{})\b".format("|".join(re.escape(kw)
                                                    for kw in sequence))
            matches = list(re.finditer(pattern, chunk))
            for i in range(len(matches) - len(sequence) + 1):
                seq_matches = matches[i:i + len(sequence)]
                if all(seq_matches[j].group() == sequence[j] for j in range(len(sequence))):
                    count += len(sequence)
    return count


def chunk_text(text, keywords, max_distance):
    """ Extracts chunks from the text based on the positions and clusters. The chunks are sorted by the number of keywords they contain so that the most relevant chunks are first.

    Args:
        text (str): The text to search.
        keywords (list): A list of keywords to search for. A keyword can be a string or a dict. If a dict, it must have a "sequence" key and an optional "max_distance" key.
        max_distance (int): The maximum distance between two positions to be considered in the same cluster.

    Returns:
        list: A list of chunks sorted by the number of keywords they contain.
    """
    positions = extract_keyword_positions(text, keywords)
    clusters = cluster_positions(positions, max_distance)
    chunks = extract_chunks(text, positions, clusters)
    sorted_chunks = sorted(chunks, key=lambda chunk: count_keywords(
        chunk, keywords), reverse=True)
    return sorted_chunks


# Example usage
# text = "This is a sample text with some keywords and ordered sequences. The proximity of keywords is important for chunks."
# keywords = ["This", "text", "sample",  {"sequence": [
#     "ordered", "sequences"], "max_distance": 100}]
# max_distance = 500

# result = chunk_text(text, keywords, max_distance)
# print(result)

# print(cluster_positions([(0, 'I'), (7, 'apples'), (18, 'oranges')], 50))
# print(extract_chunks("I like apples and oranges", [
#       (0, 'I'), (7, 'apples'), (18, 'oranges')], [0, 0, 0]))
