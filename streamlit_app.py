import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassThrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


from langchain_community.document_loaders import TextLoader
loader_customer = TextLoader("F:/Downloads/Customer Feedback Guidelines.txt")
docs_customer = loader_customer.load()

loader_company = TextLoader("F:/Downloads/Responsibility_distribution.txt")
docs_company = loader_company.load()

text_splitter1 = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunk_documents1 = text_splitter1.split_documents(docs_customer)

text_splitter2 = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunk_documents2 = text_splitter2.split_documents(docs_company)


prompt1 = ChatPromptTemplate.from_template(""""
        Act as a customer service chatbot, think step by step before providing answers. Be humble and friendly.
        <context>
        {context}
        </context>

        Question: {input}""")

prompt2 = ChatPromptTemplate.from_template(""""
        You are a chatbot for the company. Chat history will be saved. Main task of this chatbot is to detect which department should deal with the data and provide feedbacks.
        <context>
        {context}
        </context>

        Question: {input}""")
    
db1 = FAISS.from_documents(chunk_documents1, OllamaEmbeddings())
db2 = FAISS.from_documents(chunk_documents2, OllamaEmbeddings())

llm = Ollama(model = "llama3.2:3b" )

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
document_chain1 = create_stuff_documents_chain(llm,prompt1)


document_chain2 = create_stuff_documents_chain(llm,prompt2)


retriever1 = db1.as_retriever()
retriever2 = db2.as_retriever()




from langchain.chains import create_retrieval_chain
retrieval_chain1 = create_retrieval_chain(retriever1, document_chain1)
retrieval_chain2 = create_retrieval_chain(retriever2, document_chain2)




def process_input_customer(feedbacks):
      return retrieval_chain1.invoke({"input":feedbacks})['answer']
def process_input_company(feedbacks):
      return retrieval_chain2.invoke({"input":feedbacks})['answer']



#streamlit UI
st.title = st.title("INSIGHT Chatbot")
st.write("Provide feedback about the new products.")
feedbacks = st.text_input("Feedback")
if st.button('Send'):
        with st.spinner('Processing...'):
            st.text_area["Reply",value = process_input_customer(feedbacks), height= 300, disabled = True ]

        prompt1 = ChatPromptTemplate.from_template(""""
        Act as a customer service chatbot, think step by step before providing answers. Be humble and friendly.
        <context>
        {context}
        </context>

        Question: {input}""")

        prompt2 = ChatPromptTemplate.from_template(""""
        You are a chatbot for the company. Chat history will be saved. Main task of this chatbot is to detect which department should deal with the data and provide feedbacks.
        <context>
        {context}
        </context>

        Question: {input}""")
    
