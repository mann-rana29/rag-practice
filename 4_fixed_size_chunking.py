def chunk_text(text,chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")

    return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

if __name__ == "__main__":
    print(chunk_text("ABCDEFGHIJKL",4))