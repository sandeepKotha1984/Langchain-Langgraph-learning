from langchain_community.document_loaders import Docx2txtLoader


def load_document(file_path):
    """
    Load a document from the given file path using Docx2txtLoader.

    Args:
        file_path (str): The path to the document file.

    Returns:
        str: The loaded document content.
    """
    loader = Docx2txtLoader(file_path)
    document = loader.load()
    return document