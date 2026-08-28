def chunk_text(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    return [text[i : i+chunk_size] for i in range(0, len(text), chunk_size-overlap) if text[i : i + chunk_size ]]

if __name__ == "__main__":
    print(chunk_text("ABCDEFGHIJKL",6,2))