def build_prompt(query, retrieved_chunks):
    context_parts=[]

    for i, chunk in enumerate(retrieved_chunks, start =1):
        source = chunk.get("metadata", {}).get("source","unknown")
        context_parts.append(
            f"[Context {i} | source={source}] \n {chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""You are a question-answering assistant. Use ONLY the context below to answer the question. If the context does not contain enough information, say: "I don't have enough information in the provided context." Do not invent facts.
    CONTEXT : {context}
    
    QUESTION : {query}

    ANSWER :
    """

print(build_prompt("who is mogambo", [{"text" : "mogambo khush hua","metadata" : { "source" : "chandigarh"}}, {"text" : "mogambo dukhi hua" , "source" : "delhi"}]))
