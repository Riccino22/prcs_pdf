import streamlit as st

def main():
    st.set_page_config(page_title="Chat with a PDF", page_icon=":books:")
    
    st.header("Chat with a multples PDFs :books:")
    st.text_input("Ask a question about your documents")
    
    with st.sidebar:
        st.subheader("Your document")
        st.file_uploader("Upload yor PDF file", type="pdf")
        st.button("Process")


if __name__ == "__main__":
    main()