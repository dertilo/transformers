import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from examples.summarization.bart.finetune import SummarizationTrainer
from transformers import BartForConditionalGeneration, BartTokenizer


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def generate_write_summaries(examples: list,out_file:str, model_name: str, batch_size: int = 8, device: str = DEFAULT_DEVICE):
    fout = Path(out_file).open("w")
    for s in generate_summaries(examples,model_name,batch_size,device):
        fout.write(s + "\n")
        fout.flush()


def generate_summaries(
    examples: list, model_name: str, batch_size: int = 8, device: str = DEFAULT_DEVICE
):

    if model_name.endswith('.ckpt'):
        model = SummarizationTrainer.load_from_checkpoint(model_name).model.to(device)
    else:
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    tokenizer = BartTokenizer.from_pretrained("bart-large")

    max_length = 140
    min_length = 55

    for batch in tqdm(list(chunks(examples, batch_size))):
        dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_beams=4,
            length_penalty=2.0,
            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=model.config.eos_token_id,
        )
        for g in summaries:
            yield tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path", type=str, help="like cnn_dm/test.source",default=os.environ['HOME']+"/hpc/data/cnn_dm/val.source",required=False
    )
    parser.add_argument(
        "--output_path", type=str, help="where to save summaries",default='./summaries.txt',required=False
    )
    parser.add_argument(
        "--model_name", type=str, default="/home/tilo/data/bart_coqa_seq2seq/checkpointepoch=0.ckpt", help="like bart-large-cnn",required=False
    )
    parser.add_argument(
        "--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.",
    )
    parser.add_argument(
        "--bs", type=int, default=1, required=False, help="batch size: how many to summarize at a time",
    )
    args = parser.parse_args()
    examples = [" " + x.rstrip() for x in open(args.source_path).readlines()]
    generate_write_summaries(examples, args.output_path, args.model_name, batch_size=args.bs, device=args.device)


if __name__ == "__main__":
    run_generate()
