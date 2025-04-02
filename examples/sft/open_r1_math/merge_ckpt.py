from transformers import AutoTokenizer, AutoModelForCausalLM

import torch 

model_paths = [
    "/home/slurm/tuenv2/tuenv/exp_r1/baseline_8192_extend_rope/global_step_7000",
    "/home/slurm/tuenv2/tuenv/exp_r1/baseline_8192_extend_rope/global_step_7500",
    "/home/slurm/tuenv2/tuenv/exp_r1/baseline_8192_extend_rope/global_step_8000",
    "/home/slurm/tuenv2/tuenv/exp_r1/baseline_8192_extend_rope/global_step_8500",
    "/home/slurm/tuenv2/tuenv/exp_r1/baseline_8192_extend_rope/global_step_9000",
    "/home/slurm/tuenv2/tuenv/exp_r1/baseline_8192_extend_rope/global_step_9500"
]

tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
model = AutoModelForCausalLM.from_pretrained(model_paths[0])
state_dict = model.state_dict()
state_dict_new = {
    k: [v] for k, v in state_dict.items()
}
for path in model_paths[1:]:
    # ckpt = torch.load(path)
    model_current = AutoModelForCausalLM.from_pretrained(path)
    ckpt = model_current.state_dict()
    for key in ckpt.keys():
        state_dict_new[key].append(ckpt[key])
for k in list(state_dict_new.keys()):
    dtype = state_dict_new[k][0].dtype
    state_dict_new[k] = torch.stack(state_dict_new[k], dim=0).mean(dim=0).to(dtype)

model.load_state_dict(state_dict_new)
model.save_pretrained("/home/slurm/tuenv2/tuenv/exp_r1/baseline_8192_extend_rope/global_step_avg_7500_9500")
tokenizer.save_pretrained("/home/slurm/tuenv2/tuenv/exp_r1/baseline_8192_extend_rope/global_step_avg_7500_9500")
