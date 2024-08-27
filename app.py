import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def main():
    st.set_page_config(page_title="Chat with a PDF", page_icon=":books:")
    
    st.header("Chat with a multples PDFs :books:")
    st.text_input("Ask a question about your documents")
    
    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader("Upload yor PDF file", type="pdf", accept_multiple_files=True)
        print(pdf_docs)
        process = st.button("Process")
        if process:
            with st.spinner("Processing"):
                raw_text =  get_pdf_text(pdf_docs)
                st.write(raw_text)


if __name__ == "__main__":
    main()