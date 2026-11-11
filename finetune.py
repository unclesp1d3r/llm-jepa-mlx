"""LLM-JEPA fine-tuning on Apple Silicon (MLX port of upstream finetune.py)."""

import argparse
import math
import os
import random

import mlx.core as mx
import mlx.optimizers as optim
from tqdm import tqdm

import jepa
import model as model_lib
from data import MAX_PREDICTORS, load_jsonl, prepare_example

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-bf16"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with the LLM-JEPA loss (MLX)")
    parser.add_argument("--train_file", required=True, help="Training JSONL file")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="mlx-lm model repo or path (unquantized)")
    parser.add_argument("--output_dir", default="./fine-tuned",
                        help="Output directory; the latest model is saved here every epoch")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=10, help="Log every this many steps")
    parser.add_argument("--keep_epochs", action="store_true",
                        help="Also keep each epoch's model in an epoch-N folder inside --output_dir")
    parser.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--finetune_seed", type=int, default=42)
    parser.add_argument("--predictors", type=int, default=0, help="Number of [PRED] tokens (k)")
    parser.add_argument("--lbd", type=float, default=0.1, help="Lambda: weight of the JEPA loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma: weight of the LLM loss")
    parser.add_argument("--last_token", type=int, default=-1,
                        help="Embedding position relative to the view's end; -1 is the final token")
    parser.add_argument("--regular", action="store_true", help="Plain next-token fine-tuning (baseline)")
    parser.add_argument("--pretrain", action="store_true", help="Train from randomly initialized weights")
    parser.add_argument("--train_all", action="store_true", help="Compute the LLM loss on all tokens")
    parser.add_argument("--plain", action="store_true",
                        help="No chat template: join query and answer with <|perception|> "
                             "(with --train_all, train on the query only)")
    parser.add_argument("--front_pred", action="store_true", help="Prepend [PRED] tokens to Text")
    parser.add_argument("--reverse_pred", action="store_true", help="Let Code predict Text")
    parser.add_argument("--jepa_ratio", type=float, default=-1.0,
                        help="When >0, apply the JEPA loss to this fraction of batches (LD = 1 - ratio)")
    parser.add_argument("--unmask_assistant_special_tokens", action="store_true",
                        help="Also train on the assistant's closing special tokens")
    loss = parser.add_mutually_exclusive_group()
    loss.add_argument("--jepa_l2", action="store_const", dest="jepa_kind", const="l2")
    loss.add_argument("--jepa_mse", action="store_const", dest="jepa_kind", const="mse")
    loss.add_argument("--infonce", action="store_const", dest="jepa_kind", const="infonce")
    parser.set_defaults(jepa_kind="cosine")

    args = parser.parse_args(argv)
    if not 0 <= args.predictors <= MAX_PREDICTORS:
        parser.error(f"--predictors must be between 0 and {MAX_PREDICTORS}")
    if min(args.batch_size, args.grad_accum, args.num_epochs) < 1:
        parser.error("--batch_size, --grad_accum and --num_epochs must be at least 1")
    return args


def prepare_dataset(args, tokenizer):
    rows = load_jsonl(args.train_file)
    use_view_fields = os.path.basename(args.train_file).startswith("hellaswag")
    return [
        prepare_example(
            row, tokenizer, model_name=args.model_name, max_length=args.max_length,
            predictors=args.predictors, regular=args.regular, train_all=args.train_all,
            plain=args.plain, front_pred=args.front_pred, reverse_pred=args.reverse_pred,
            unmask_assistant_special_tokens=args.unmask_assistant_special_tokens,
            use_view_fields=use_view_fields,
        )
        for row in tqdm(rows, desc="Tokenizing")
    ]


def count_at_limit(examples, max_length):
    lengths = ("input_ids", "text_ids", "code_ids")
    return sum(
        any(len(example[name]) >= max_length for name in lengths if name in example)
        for example in examples
    )


def train(args, model, examples, on_epoch_end):
    steps_per_epoch = len(examples) // args.batch_size  # drops the last partial batch
    # like transformers.Trainer, a partial accumulation group is applied at each epoch's end
    updates = math.ceil(steps_per_epoch / args.grad_accum) * args.num_epochs
    if updates < 1:
        raise ValueError(f"{len(examples)} examples is not enough for a single optimizer update.")
    # transformers.Trainer defaults: AdamW, no weight decay, linear decay to zero
    optimizer = optim.AdamW(
        learning_rate=optim.linear_schedule(args.learning_rate, 0.0, updates),
        weight_decay=0.0, bias_correction=True,
    )
    step = jepa.make_train_step(
        model, optimizer, gamma=args.gamma, lbd=args.lbd, kind=args.jepa_kind,
        grad_accum=args.grad_accum,
    )
    rng = random.Random(args.finetune_seed)
    model.train()
    for epoch in range(args.num_epochs):
        order = rng.sample(range(len(examples)), len(examples))
        for index in range(steps_per_epoch):
            rows = order[index * args.batch_size:(index + 1) * args.batch_size]
            batch = jepa.collate([examples[row] for row in rows], args.last_token)
            use_jepa = not args.regular and (args.jepa_ratio <= 0 or rng.random() <= args.jepa_ratio)
            metrics = step(batch, use_jepa, flush=index == steps_per_epoch - 1)
            if (index + 1) % args.eval_steps == 0:
                print(f"epoch {epoch + 1} step {index + 1}/{steps_per_epoch} "
                      f"loss {metrics['loss']:.4f} llm_loss {metrics['lm']:.4f} "
                      f"jepa_loss {metrics['jepa']:.4f}", flush=True)
        on_epoch_end(epoch + 1)
    model.eval()
    return int(optimizer.step.item())


def main(argv=None):
    args = parse_args(argv)
    mx.random.seed(args.finetune_seed)

    print(f"Model: {args.model_name}\nTrain file: {args.train_file}\nLoRA: {args.lora}")
    model, tokenizer, config = model_lib.load(args.model_name, pretrain=args.pretrain)
    if args.lora:
        model_lib.apply_lora(model, args.lora_rank)
    examples = prepare_dataset(args, tokenizer)
    print(f"Train samples: {len(examples)}")
    if not args.regular:
        jepa.check_view_positions(examples, args.last_token)
    print(f"Examples at the --max_length limit of {args.max_length} (cut, or exactly that long): "
          f"{count_at_limit(examples, args.max_length)}")

    def save_epoch(epoch):
        model_lib.save(
            args.output_dir, model, tokenizer, config, epoch=epoch, keep_epoch=args.keep_epochs
        )
        print(f"Epoch {epoch}: model saved to {args.output_dir}", flush=True)

    train(args, model, examples, save_epoch)
    print(f"Training completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
