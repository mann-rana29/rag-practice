from pathlib import Path

def load_docuemnt(path):
    return Path(path).read_text(encoding='utf-8')

if __name__ == "__main__":
    print(load_docuemnt("data/file.txt"))