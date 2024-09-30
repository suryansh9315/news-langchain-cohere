import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool")

st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

query = st.text_input("Question: ")
process_button = st.button("Process URLs and Generate")

if process_button:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    r_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    chunks = r_splitter.split_documents(data)
    chunks_string = []
    for chunk in chunks:
        chunks_string.append(chunk.page_content)

    embeddings_model = CohereEmbeddings(model='embed-english-v3.0')
    vectorstore = FAISS.from_documents(chunks, embeddings_model)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatCohere(model="command-r-plus")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    st.header("Answer")
    st.write(response['answer'])

    sources = []
    for doc in response.get('context'):
        sources.append(doc.metadata['source'])
    if sources:
        st.subheader("Sources")
        for source in sources:
            st.write(source)