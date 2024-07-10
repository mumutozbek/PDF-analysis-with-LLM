# PDF Analysis with Llama3

This project provides a Streamlit app for analyzing PDF files using Llama3.

## Features

- Upload a PDF file and process it
- Summarize and analyze the content of the PDF
- Ask questions based on the content of the PDF

## Setup

### Option 1: Use Pre-configured Virtual Environment

1. Clone the repository:
    ```bash
    git clone https://github.com/mumutozbek/PDF-analysis-with-LLM.git
    cd your-repo-name
    ```

2. Download the virtual environment:
    ```bash
    # For myenv.tar.gz
    tar -xzvf myenv.tar.gz

    # For myenv.zip
    unzip myenv.zip
    ```

3. Activate the virtual environment:
    ```bash
    # On Windows
    myenv\Scripts\activate

    # On macOS/Linux
    source myenv/bin/activate
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

### Option 2: Set Up Your Own Virtual Environment

1. Clone the repository:
    ```bash
    git clone https://github.com/mumutozbek/PDF-analysis-with-LLM.git
    cd your-repo-name
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the Streamlit app in your browser.
2. Upload a PDF file.
3. View the summary and analysis.
4. Ask questions based on the content of the PDF.
