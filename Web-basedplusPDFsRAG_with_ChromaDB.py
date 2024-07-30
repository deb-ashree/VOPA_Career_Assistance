# pip install langchain langchainhub langchain_objectbox langchain_community

## Krish Naik - https://github.com/krishnaik06/Updated-Langchain/blob/main/openai/GPT4o_Lanchain_RAG.ipynb, https://www.youtube.com/watch?v=TcvI-Nnebow
## https://github.com/NebeyouMusie/End-to-End-RAG-Project-using-ObjectBox-and-LangChain implemented on krish Naik's learnings

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma ##vector Database
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings

load_dotenv(os.getcwd()+"/local.env")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['USER_AGENT'] = 'test_agent'

ef_openai = OpenAIEmbeddings()

# load from disk
vector = Chroma(persist_directory="chromaDb", embedding_function=ef_openai)
if vector is None:
    # 1. Get a Data Loader

    print("Loading from the Web")
    # List of URLs to load documents from
    urls = [
        # "https://www.careerguidancejpgandhi.com/",
        # "https://www.mindler.com/blog/medical-courses-after-12th/",
        # "https://www.joinindiannavy.gov.in/en/page/medical.html",
        # "https://www.shiksha.com/engineering-career-chp",
        "https://www.careers360.com",
        "https://collegedunia.com/",
        # "https://gettinggrowth.com/career-in-marketing/",
        # "https://edufever.in/career-in-journalism-in-india/#:~:text=to%20achieve%20this.-,Job%20Opportunities%20for%20Journalism,a%20degree%20in%20broadcast%20journalism.",
        # "https://edufever.in/category/courses/",
        # "https://maharashtracareerportal.com/#googtrans(en)",
    ]

    # Load documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs_list)
    documents

    # 2. Convert data to Vector Database

    vector = Chroma.from_documents(collection_name="test_chroma_db", documents=documents, embedding=ef_openai, persist_directory='chromaDb')
    print(vector)

    # 3. Repeat for PDFs

    print("Loading from the PDF")
    path_to_pdfs = 'pdf_files/'
    if os.path.exists(path_to_pdfs):
        file_list = os.listdir(path_to_pdfs)
        for file in file_list:
            file = os.path.join(path_to_pdfs,file)
            print(file)
            loader = PyPDFLoader(file)
            data = loader.load()
            print("here++++")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            pdf_splits = text_splitter.split_documents(data)

    # 4. Convert data to Vector Database
    ##vector = ObjectBox.from_documents(pdf_splits, OpenAIEmbeddings(), embedding_dimensions=768)
    vector.add_documents(pdf_splits, embedding_dimensions=768)
    print(vector)

# 5. Make a RAG pipeline

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub

llm_gpt = ChatOpenAI(model="gpt-4o-mini") ## Calling Gpt-4o
prompt = hub.pull("rlm/rag-prompt")
print(prompt)

llm_ollama = ChatOllama(
    model="llama3.1",
    temperature=.7,
)

## Used to verify content
# pvector= vdatabase.get() 
# docs = pvector.similarity_search(query)
# print(docs[0].page_content)

qa_chain = RetrievalQA.from_chain_type(
        llm_gpt,
        retriever=vector.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

question = "What are the career options with Arts" # "How do I get into Engineering" "How do I get into journalism"
result = qa_chain.invoke({"query": question })
#print(result)

import pprint
pp = pprint.PrettyPrinter(indent=5)
pp.pprint(result["result"])
