import re

def preprocess_and_split(text : str, chunk_size : int = 100, overlap : int= 20) -> list:
    cleaned_text = re.sub(r'\s+', '', text).strip()

    chunks = []

    start = 0

    text_length = len(cleaned_text)
    while start < text_length:
        end = min(start+chunk_size, text_length)

        chunks.append(cleaned_text[start:end])

        start = start + chunk_size - overlap

    return chunks


raw_doc = "Retrieval-Augmented Generation combines external data with LLM generation. It improves accuracy."
print(preprocess_and_split(raw_doc, chunk_size=40, overlap=10))