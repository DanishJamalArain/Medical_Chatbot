from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


PINECONE_API_KEY ="pcsk_78DN3L_DoHkTvuX5AarJBbp929P7tQdTeJERVAz68sTW5R6F3AHoScCuEqzHyXBsk1mUrw"
OPENAI_API_KEY ="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"

#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings