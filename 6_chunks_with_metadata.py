def build_chunks(chunks,source):
    records = []

    for index, chunk in enumerate(chunks):
        records.append({
            "id" : f"doc_0_chunk_{index}",
            "text" : chunk,
            "metadata":{
                "source" : source,
                "chunk_index" : index
            }
        })


    return records

if __name__ == "__main__":
    chunks = ["first chunk", "second chunk"]
    print(build_chunks(chunks, "sample.txt"))
