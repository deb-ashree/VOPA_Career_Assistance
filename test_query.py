from RAG_document_loader import LoadContextData
if __name__ == "__main__":
    # old_stdout = sys.stdout
    # log_file = open("message.log","w")
    # sys.stdout = log_file
    
    context = LoadContextData()
    #  context.checkCollection("test_chroma_db")
    #  context.checkCollection("test_db")
    # print("Load Data..")
    # vector = context.loadAndStoreFiles()
    # print("Loading Data Done..")
    language = "english"
    question = "Who is Firdose Kapadia?"   #  "How do I get into Engineering" "How do I get into journalism"
    result = context.queryDataFromVector(context.vector, question, language)
    print("Response from Context..")

    vector = context.getVector()
    # res = vector._chroma_collection(query_texts="Who is Firdose Kapadia?", n_results=2)
    # res = vector._collection.query(query_texts="Who is Firdose Kapadia?", n_results=2)
    results = vector.similarity_search(
        # "An international trainer and a Henry Giessenbier Fellow and Senator of JCI-USA",
        "Henry Giessenbier",
        k=1,
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
    # sys.stdout = old_stdout
    # log_file.close()