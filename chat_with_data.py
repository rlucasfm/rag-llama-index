from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, set_global_tokenizer
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from transformers import AutoTokenizer
from timeit import default_timer as timer

start = timer()

llm = LlamaCPP(
    # model_url=model_url,
    model_path="../Meta-Llama-3-8B-Instruct.Q2_K.gguf",
    temperature=0.3,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 60},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# # Para fazer completions direto do modelo
# response = llm.complete("Hello! Make me a cronogram for a trumpet weekly studies?")
# print(response.text)

# Para fazer buscas RAG
set_global_tokenizer(
    AutoTokenizer.from_pretrained("../Meta-Llama-3-8B").encode
)

embed_model = HuggingFaceEmbedding(model_name="../bge-small-en-v1.5")

# # -------- Criar um VectorStore Index a partir de um documento e persistir  --------
# # load documents
# documents = SimpleDirectoryReader(
#     "./data"
# ).load_data()

# # create vector store index
# index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# index.storage_context.persist(persist_dir="vectorstorage")
# # -----------------------------------------------------------------------------------------

# -------- Carregar um VectorStore Index a partir de um index salvo anteriormente --------
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="vectorstorage")
index = load_index_from_storage(storage_context, embed_model=embed_model)
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# # Retriever para testar, caso queira apenas pegar os dados da Index Storage
# retriever = index.as_retriever()
# nodes = retriever.retrieve('Em quanto tempo serão desenvolvidas as etapas do cronograma físico-financeiro?')
# for node in nodes:
#     print("----------------\n")
#     print(node.text)
#     print("----------------\n")
# -----------------------------------------------------------------------------------------

# set up query engine
query_engine = index.as_query_engine(llm=llm, streaming=True)

# Query for a response
response = query_engine.query("Responda somente em português. Qual a Situação Atual da Área de Implantação do Empreendimento??")
response.print_response_stream()

# ------- Get response Metadata -------
if hasattr(response, 'metadata'):
    print(response.metadata)

end = timer()
print('Execution time: {}'.format(end - start))