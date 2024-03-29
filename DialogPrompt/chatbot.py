# DialogPrompt
# Copyright 2022-present NAVER Corp.
# BSD 3-clause

import os
import logging
from models.dialogprompt import PromptEncoder
import torch
#try:
#    from torch.utils.tensorboard import SummaryWriter
#except:
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm, trange

import models
from models import DialogPrompt
from data_loader import DialogTransformerDataset, ResponseGPTDataset, DialogGPTDataset
from learner import Learner
from transformers import (GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config)
from modules import MaskableLSTM
import argparse
def main():
    
    parser = argparse.ArgumentParser(description="Arguments for your program")

    # Define arguments with types and descriptions
    parser.add_argument("--model_size", default="base", type=str, help="base, medium, large")
    parser.add_argument("--language", default="english", type=str, help= "language, english or chinese")
    parser.add_argument('--num_trig_tokens', type=float, default=20)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--fast_eval_ratio", default=0.1, type=float, help="Ratio of samples for fast test in the test set. dailydial: 1.0")

    # Parse arguments and store them in args object
    args = parser.parse_args()
    args.num_trig_tokens = int(args.num_trig_tokens)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = './output/DialogPrompt/models/checkpoint-optimal'
    args = torch.load(os.path.join(output_dir, 'training_args.bin'))
    model = DialogPrompt(args)

    model.load_state_dict(torch.load(os.path.join(output_dir, 'state_dict.bin'), map_location=torch.device('cpu')), strict=False)
    model.eval()

    
        
    context = []

    while True:
        user_input = input('You:')
        context.append(user_input)
        # context = " ".join(conversation_history)
        test_set = ResponseGPTDataset(context, model.tokenizer)
        loss, result, generated_text, pred_sents = Learner().run_eval(args, model, test_set)
        print("Chatbot:", pred_sents)
        context.append(pred_sents)
        if len(context) > 2:
            context = context[-2:]


if __name__ == "__main__":
    main()
