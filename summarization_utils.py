import webvtt
from io import StringIO
import os

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


def configure_map_reduce_chain():
    """
    Configure a map-reduce chain for summarization.
    """

    llm = Ollama(model="llama3", temperature=0)

    # Map
    map_template = """
    The following is a set of meeting documents:
    {docs}

    Based on these documents, please identify the main themes discussed. Focus on extracting key topics, major decisions made, and any critical action items. Present these themes as a bulleted list to ensure clarity and conciseness.

    Helpful Answer:
    """

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """
    The following is a set of summaries from multiple meetings:
    {docs}

    Please synthesize these summaries into a final, consolidated summary. Focus on integrating the main themes into a coherent narrative. Structure the final summary into sections based on different themes or topics. Ensure the summary is concise, capturing only the most essential and relevant points.

    Helpful Answer:
    """

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combining and iteratively reducing the mapped documents
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )
    return map_reduce_chain


def parse_vtt(file_content: str) -> str:
    """
    Parses VTT content and returns plain text.

    Args:
        file_content (str): The VTT file content.

    Returns:
        str: The plain text.
    """

    vtt = webvtt.read_buffer(StringIO(file_content))
    lines = [caption.text.strip() for caption in vtt]
    return " ".join(lines)


def process_text(text: str) -> str:
    """
    Process text for summarization using a map reduce chain.

    Args:
        text (str): The text to summarize.

    Returns:
        str: The summarized text.

    """

    map_reduce_chain = configure_map_reduce_chain()
    # Split text into manageable parts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    split_docs = text_splitter.split_text(text)
    split_docs = text_splitter.create_documents(split_docs)
    # Apply the map-reduce summarization
    summary = map_reduce_chain.run(split_docs)
    return summary


def summarize_file(file_path: str) -> str:
    """
    Summarize a file for summarization using a map reduce chain.

    Args:
        file_path (str): The path to the file to summarize.

    Returns:
        str: The summarized text.

    """

    # Determine the file extension
    _, file_extension = os.path.splitext(file_path)
    text = ""

    if file_extension.lower() == ".vtt":
        # If the file is a VTT file, parse it to extract text
        for caption in webvtt.read(file_path):
            text += caption.text + " "
    else:
        # Assume it's a plain text file
        with open(file_path, "r") as file:
            text = file.read()

    return process_text(text)
