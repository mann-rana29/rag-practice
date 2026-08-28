import re

def preprocess(text):
    text = text.lower()

    text = re.sub(r"\s+", " ", text) #\s tells everything that is space , new lines or tabs..   + tells more than once in a line..

    return text.strip()

if __name__ == "__main__":
    raw = "  Hello  WORLD\n\nThis\tis RAG."
    print(preprocess(raw))