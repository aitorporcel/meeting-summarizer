import gradio as gr
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import TextLoader

llm = Ollama(model="llama3", temperature=0)

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
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


def process_text(text):
    # Split text into manageable parts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    split_docs = text_splitter.split_text(text)
    split_docs = text_splitter.create_documents(split_docs)
    # Apply the map-reduce summarization
    summary = map_reduce_chain.run(split_docs)
    return summary


def summarize_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return process_text(text)


with gr.Blocks() as app:
    gr.Markdown("### Meeting Transcription Summarizer")
    gr.Markdown(
        "Upload a meeting transcription file or paste text directly, and this app will use a map-reduce strategy to summarize it."
    )

    with gr.Tab("Upload File"):
        file_input = gr.File(file_types=["txt"], label="Upload your transcription file")
        file_output = gr.Textbox(label="Summary")
        file_button = gr.Button("Summarize File")
        file_button.click(fn=summarize_file, inputs=file_input, outputs=file_output)

    with gr.Tab("Enter Text"):
        text_input = gr.Textbox(
            label="Paste text here", lines=10, placeholder="Enter text here..."
        )
        text_output = gr.Textbox(label="Summary")
        text_button = gr.Button("Summarize Text")
        text_button.click(fn=process_text, inputs=text_input, outputs=text_output)

    app.launch()
