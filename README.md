# Meeting Transcription Summarizer

The Meeting Transcription Summarizer is a local, privacy-focused tool designed to quickly and effectively summarize meeting transcriptions. Leveraging open-source Large Language Models (LLMs) like Llama 3 7B using Ollama, this tool performs a map-reduce strategy to digest and condense the content of meeting documents, ensuring all data processing remains on the local machine to address privacy concerns.

## Features

- **Local Processing**: Ensures that all data remains on your local machine, avoiding potential privacy issues associated with cloud processing.
- **Support for Multiple Formats**: Processes both plain text and VTT files, catering to a variety of transcription outputs, particularly from platforms like Zoom.
- **Map-Reduce Summarization**: Utilizes a map-reduce approach to handle extensive documents and extract concise summaries focusing on key themes, decisions, and action items.
- **Easy-to-Use Interface**: Offers a simple web interface built with Gradio, allowing users to upload files or paste text directly for summarization.

## Preliminary Setup

Before you can fully utilize the Meeting Transcription Summarizer, you need to set up the underlying Large Language Model (LLM), using Ollama, and download the model you want to use, for example Llama 3 7B model. Follow these steps to prepare your environment:

Download and Install Ollama:
Ollama is an open-source LLM framework that supports various models. Ensure you have Ollama installed in your computer and enviroment.
To install Ollama you can visit their website: [https://ollama.com/download](https://ollama.com/download)

Download the model to be used:
The application uses a model to process and summarize transcriptions. You need to download this model to your local machine to use it. For example you can download the llama3 model using:

```bash
ollama pull llama3
```

Note: If you want to use a different model you have to update the model name in the `app.py` file.

## Installation

To set up the Meeting Transcription Summarizer on your local machine, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/meeting-transcription-summarizer.git
   cd meeting-transcription-summarizer
   ```

2. **Install dependencies**:
   Ensure you have Python installed, and then install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the application, run the following command in the terminal:

```bash
python app.py
```

This will launch a web interface accessible at `http://localhost:7860` by default, where you can:

- **Upload a file**: Supported file types include `.txt` and `.vtt`. Click the "Summarize File" button after uploading your file.
- **Enter text directly**: Paste the text into the textbox and click the "Summarize Text" button to get a summary.

## Contributing

Contributions to enhance or expand the capabilities of the Meeting Transcription Summarizer are welcome. Please fork the repository and submit a pull request with your enhancements. You can also open issues for bugs or feature requests.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or feedback, please don't hesitate to reach out via GitHub issues or pull requests.
