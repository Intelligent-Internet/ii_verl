from verl import DataProto
from rl_verifier import RLVerifierClient
import torch
import os
import pandas as pd

class APIRewardManager():
    """Reward model as API.
    """

    def __init__(self, tokenizer, api_url, max_workers = 10, timeout = 30, verification_info_column = 'verification_info', save_dir = "debug_data", save_freq = 1):
        self.tokenizer = tokenizer
        self.client = RLVerifierClient(api_url, timeout=timeout)
        self.max_workers = max_workers
        self.verification_info_column = verification_info_column
        
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.step = 0
        
    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        lst_request_items = []
        lst_score_positions = []
        lst_ids = []
        lst_data_sources = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            llm_output = self.tokenizer.decode(sequences)

            verification_info = data_item.non_tensor_batch.get(self.verification_info_column)
            if not verification_info:
                raise ValueError(f"Verification info column {self.verification_info_column} not found in data")
            
            idx = data_item.non_tensor_batch.get('id')
            data_source = data_item.non_tensor_batch.get('data_source')
            if idx:
                lst_ids.append(idx)
            if data_source:
                lst_data_sources.append(data_source)
            
            lst_request_items.append(
                (llm_output, verification_info)
            )
            lst_score_positions.append(valid_response_length - 1)

        # Call API to get reward scores
        scores = self.client.verify_batch(
            batch=lst_request_items,
            max_workers=self.max_workers,
            default_value=0.0,
            progress_bar=True
        )
        
        print("**Accuracy**:", sum(scores) / len(scores))
        
        # ------------------------------------------------------------
        # Save the data for debugging
        if self.save_freq > 0 and self.step % self.save_freq == 0:
            lst_llm_outputs = [item[0] for item in lst_request_items]
            lst_verification_info = [item[1] for item in lst_request_items]
            data_dict = {
                'llm_output': lst_llm_outputs,
                'verification_info': lst_verification_info,
                'score': scores
            }
            if lst_ids:
                data_dict['id'] = lst_ids
            if lst_data_sources:
                data_dict['data_source'] = lst_data_sources
            df = pd.DataFrame(data_dict)
            df.to_parquet(os.path.join(self.save_dir, f"{self.step}.parquet"))
        self.step += 1
        # ------------------------------------------------------------
        
        for i, score in enumerate(scores):
            reward_tensor[i, lst_score_positions[i]] = score

        return reward_tensor