from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
import os
import torch
print("======================================")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("======================================")

os.environ["HF_TOKEN"] = " [REMOVED]"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("======================================")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("======================================")

# 1. Load and split document
loader = TextLoader("../inputTextFiles/unsolicitated_advice.txt")
documents = loader.load()
# splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=50)
splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100)
docs = splitter.split_documents(documents)

print("======================================")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("======================================")

# 2. Embed documents
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = FAISS.from_documents(docs, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# retriever = db.as_retriever(search_kwargs={"k": 1})

print("======================================")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("======================================")

# 3. Load GPTQ LLM
model_name = "thesven/Mistral-7B-Instruct-v0.3-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

print("======================================")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("======================================")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
llm = HuggingFacePipeline(pipeline=pipe)

print("======================================")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("======================================")

# 4. QA Chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

print("======================================")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("======================================")

# 5. Query
query = "Explain what are the 3 big ideas from this text and explain each of them in brief"
response = qa.run(query)
print("\nAnswer:\n", response)

print("======================================")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("======================================")

