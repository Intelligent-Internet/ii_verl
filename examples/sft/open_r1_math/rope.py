import math 
bash = 10000
scaling_factor = 4
seq_len = 32768 
max_position_embeddings = 4096
dim = 3584
base = bash * (
    (scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1)
) ** (dim / (dim - 2))
print(base)