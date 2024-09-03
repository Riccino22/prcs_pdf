import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

#from langchain.chains.conversational_retrieval.nase
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from html_template import css, bot_template, user_template
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
    
def get_conversation_chain(vectorestore):
        #llm = ChatGroq()
        llm = ChatGroq()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorestore.as_retriever(),
            memory=memory
        )
        return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.write(response)

def main():
    st.set_page_config(page_title="Chat with a PDF", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    
    st.header("Chat with a multples PDFs :books:")
    user_input = st.text_input("Ask a question about your documents")
    
    st.write(user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)
    if user_input:
        handle_userinput(user_input)
    
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
                
                st.session_state.conversation = get_conversation_chain(vectorstore)


                
    st.session_state.conversation 
if __name__ == "__main__":
    main()