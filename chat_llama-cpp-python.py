from llama_cpp import Llama

llm = Llama(
    model_path="../Meta-Llama-3-8B-Instruct.Q2_K.gguf",
    n_gpu_layers=20
)
llm.verbose=False

output = llm(
    "Q: Tell me what you know about global economy A:",
    max_tokens=256,
    stop=["Q:", "\n"],
    echo=False
)

print(output['choices'][0])