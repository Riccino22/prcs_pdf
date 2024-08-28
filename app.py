import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
    
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
                
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                
                vectorstore = get_vectorstore(text_chunks)


if __name__ == "__main__":
    main()