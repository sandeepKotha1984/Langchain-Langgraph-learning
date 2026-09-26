def build_prompt(query, results):
    print(f"Building prompt with query: {query} and results: {results}")
    context = "\n\n".join(
        result.content
        for result in results
    )

    return f"""
You are an enterprise policy assistant.

Answer the question using only the provided context.

Pay close attention to numerical values, ranges, currencies,
dates, thresholds, and approval levels.

Context:
{context}

Question:
{query}

Answer:
"""