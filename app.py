import gradio as gr

from summarization_utils import summarize_file, process_text

with gr.Blocks() as app:
    gr.Markdown("# Meeting Transcription Summarizer")
    gr.Markdown(
        "Upload a meeting transcription file or paste text directly, and this app will use a map-reduce strategy to summarize it."
    )

    with gr.Tab("Upload File"):
        file_input = gr.File(
            file_types=["txt", "vtt"], label="Upload your transcription file"
        )
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
