

import os
import sys
from typing import List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from utils.pdf_ocr_to_text_with_fitz import ScannedDataExtractor
load_dotenv(os.getcwd()+"/local.env")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['USER_AGENT'] = 'test_agent'

ef_openai = OpenAIEmbeddings()

class LoadContextData():
    def __init__(self):
        self.llm_gpt = ChatOpenAI(model="gpt-4o-mini") ## Calling Gpt-4o 
        self.prompt = "You are a helpful assistant who gets the available information from the context given. Only provide answer from the current context provided. Otherwise you can mention that you are capable to answer only basis the data provided"  #hub.pull("rlm/rag-prompt")

        self.ef_openai = OpenAIEmbeddings()

        self.vector = Chroma(persist_directory="chromaDB", embedding_function=self.ef_openai, collection_name="test_chroma_db_text")

    def loadText(self):

        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=200,
                        length_function=len )

        path_to_scans = 'scanned_pdf_files/'
        output_path = 'scanned_pdf_files/scans/'
        file_prefix = "image"

        file_list = os.listdir(path_to_scans)
        # scannedText = ""
        if len(file_list) > 0:
            print(len(file_list))
            extractor = ScannedDataExtractor()
            documents = extractor.extractImagesWithFitz(file_list, output_path, path_to_scans)
            print(f" Check scans : {documents}")
            # scan_text_extract = [[extractor.readScan(output_path+item) for item in listdir(output_path) if isfile(join(output_path, item))]]
            extractor.cleanUpImages(output_path)
            # if scannedText:
            #     scannedText = scannedText.replace('"',"").replace("'/","")
            #     print("Cleaned text scan ")
            #     print(scannedText)
            #     text_splitter = RecursiveCharacterTextSplitter(
            #         chunk_size=1500,
            #         chunk_overlap=200,
            #         length_function=len
            #     )
            #     scan_splits = text_splitter.split_text(scannedText)

            #     self.vector.from_texts(
            #             texts = scan_splits,
            #             embedding = self.ef_openai,
            #             collection_name="test_chroma_db_text", 
            #             persist_directory='chromaDB'
            #         )

            docsplits = text_splitter.split_documents(documents)

            # Convert data to Vector Database
            if docsplits is not []:
                self.vector.from_documents(collection_name="test_chroma_db_text", documents=docsplits, 
                                    embedding=self.ef_openai, persist_directory='chromaDB')  #(pdf_splits,embedding=self.ef_openai)
            print(self.vector)


        return self.vector
    
    def queryDataFromVector(self,vector, query, language):
        print("Creating chain")
        print(vector)
        qa_chain = RetrievalQA.from_chain_type(
                self.llm_gpt,
                retriever=vector.as_retriever(),
                #chain_type_kwargs={"prompt": self.prompt}
            )
        #question = "How is Langchain used. Respond in proper Marathi without any unicode characters" # "WHat is a Data Mesh and how is it created" "Explain Monitoring and A/B Testing in langsmith", "Who is a casual taxable person
        result = qa_chain.invoke({"query": f" {query}. Respond in the language of the query or the preferred {language} and use that language based script" + self.prompt })
        # logging.debug(result)

        import pprint
        pp = pprint.PrettyPrinter(indent=5)
        pp.pprint(result["result"])
        return result

    def getVector(self):
        return self.vector
    

if __name__ == "__main__":
    old_stdout = sys.stdout
    log_file = open("test.log","w")
    sys.stdout = log_file
    
    context = LoadContextData()

    print("Load Data..")
    vector = context.loadText()
    print("Loading Data Done..")
    language = "english"
    question = "Who is Firdose Kapadia?"   #  "How do I get into Engineering" "How do I get into journalism"
    result = context.queryDataFromVector(context.vector, question, language)
    print("Response from Context..")

    vector = context.getVector()
    # res = vector._chroma_collection(query_texts="Who is Firdose Kapadia?", n_results=2)
    # res = vector._collection.query(query_texts="Who is Firdose Kapadia?", n_results=2)
    results = vector.similarity_search(
        # "An international trainer and a Henry Giessenbier Fellow and Senator of JCI-USA",
        "Who is Firdose Kapadia",
        k=1,
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")


    sys.stdout = old_stdout
    log_file.close()
