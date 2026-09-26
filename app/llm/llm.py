from langchain_ollama import ChatOllama


def get_llm():
    llm = ChatOllama(model="qwen3:4b-instruct", temperature=0, reasoning=True);
    return llm