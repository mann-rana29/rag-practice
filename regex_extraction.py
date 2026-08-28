import re

def extract_emails(text):
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" #{2,} makes sure the pattern has atleast 2 values   [whatever in these brackets are for matching patterns like for the first one its says from A-Z , a-z , 0-9 or dots or underscores or percents or plus or hyphens match them]
    text = re.findall(pattern ,text)
    return text;

def extract_file_extension(text):
    return re.findall(r"(?i)\.[a-z0-9]+\b",text) #(?i) makes the entire pattern case insensitive so that .PDF also matches  \b is a word boundary it prevents matching middle sections and ensures the pattern stops at the end of the extension.


if __name__ == "__main__":
    text = "Contact a@x.com or b@example.org. Files: notes.txt, code.py, report.pdf abc.tar.gz"
    print(extract_emails(text))
    print(extract_file_extension(text))