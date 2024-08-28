import os
import argparse
import pandas as pd
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF
from peft import LoraConfig, get_peft_model
from IndicTransToolkit import IndicProcessor, IndicDataCollator

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)

bleu_metric = BLEU()
chrf_metric = CHRF()


def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
    )
    parser.add_argument(
        "--src_lang_list",
        type=str,
        help="comma separated list of source languages",
    )
    parser.add_argument(
        "--tgt_lang_list",
        type=str,
        help="comma separated list of target languages",
    )
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--warmup_ratio", type=int, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--print_samples", action="store_true")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch",
        choices=[
            "adam_hf",
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_apex_fused",
            "adafactor",
        ],
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="inverse_sqrt",
        choices=[
            "inverse_sqrt",
            "linear",
            "polynomial",
            "cosine",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "azure_ml", "none"],
    )
    parser.add_argument("--patience", type=int, default=5),
    parser.add_argument("--threshold", type=float, default=1e-3)
    return parser


def load_and_process_translation_dataset(
    data_dir,
    split="train",
    tokenizer=None,
    processor=None,
    src_lang_list=None,
    tgt_lang_list=None,
    num_proc=8,
    seed=42
):
    complete_dataset = {
        "sentence_SRC": [],
        "sentence_TGT": [],
    }

    for src_lang in src_lang_list:
        for tgt_lang in tgt_lang_list:
            if src_lang == tgt_lang:
                continue
            src_path = os.path.join(
                data_dir, split, f"{src_lang}-{tgt_lang}", f"{split}.{src_lang}"
            )
            tgt_path = os.path.join(
                data_dir, split, f"{src_lang}-{tgt_lang}", f"{split}.{tgt_lang}"
            )
            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                raise FileNotFoundError(
                    f"Source ({split}.{src_lang}) or Target ({split}.{tgt_lang}) file not found in {data_dir}"
                )
            with open(src_path, encoding="utf-8") as src_file, open(
                tgt_path, encoding="utf-8"
            ) as tgt_file:
                src_lines = src_file.readlines()
                tgt_lines = tgt_file.readlines()

            # Ensure both files have the same number of lines
            assert len(src_lines) == len(
                tgt_lines
            ), f"Source and Target files have different number of lines for {split}.{src_lang} and {split}.{tgt_lang}"

            complete_dataset["sentence_SRC"] += processor.preprocess_batch(
                src_lines, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False
            )

            complete_dataset["sentence_TGT"] += processor.preprocess_batch(
                tgt_lines, src_lang=tgt_lang, tgt_lang=src_lang, is_target=True
            )

    complete_dataset = Dataset.from_dict(complete_dataset).shuffle(seed=seed)

    return complete_dataset.map(
        lambda example: preprocess_fn(
            example,
            tokenizer=tokenizer
        ),
        batched=True,
        num_proc=num_proc,
    )


def compute_metrics_factory(
    tokenizer, metric_dict=None, print_samples=False, n_samples=10
):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        labels[labels == -100] = tokenizer.pad_token_id
        preds[preds == -100] = tokenizer.pad_token_id

        with tokenizer.as_target_tokenizer():
            preds = [
                x.strip()
                for x in tokenizer.batch_decode(
                    preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            ]
            labels = [
                x.strip()
                for x in tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            ]

        assert len(preds) == len(
            labels
        ), "Predictions and Labels have different lengths"

        df = pd.DataFrame({"Predictions": preds, "References": labels}).sample(
            n=n_samples
        )

        if print_samples:
            for pred, label in zip(df["Predictions"].values, df["References"].values):
                print(f" | > Prediction: {pred}")
                print(f" | > Reference: {label}\n")

        return {
            metric_name: metric.corpus_score(preds, [labels]).score
            for (metric_name, metric) in metric_dict.items()
        }

    return compute_metrics


def preprocess_fn(example, tokenizer, **kwargs):
    model_inputs = tokenizer(
        example["sentence_SRC"], truncation=True, padding=False, max_length=256
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["sentence_TGT"], truncation=True, padding=False, max_length=256
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    print(f" | > Loading {args.model} and tokenizer ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        attn_implementation="eager",
        dropout=args.dropout
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = IndicProcessor(inference=False) # pre-process before tokenization
    
    data_collator = IndicDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding="longest", # saves padding tokens
        pad_to_multiple_of=8, # better to have it as 8 when using fp16
        label_pad_token_id=-100
    )

    if args.data_dir is not None:
        train_dataset = load_and_process_translation_dataset(
            args.data_dir,
            split="train",
            tokenizer=tokenizer,
            processor=processor,
            src_lang_list=args.src_lang_list.split(","),
            tgt_lang_list=args.tgt_lang_list.split(","),
        )
        print(f" | > Loaded train dataset from {args.data_dir}. Size: {len(train_dataset)} ...")

        eval_dataset = load_and_process_translation_dataset(
            args.data_dir,
            split="dev",
            tokenizer=tokenizer,
            processor=processor,
            src_lang_list=args.src_lang_list.split(","),
            tgt_lang_list=args.tgt_lang_list.split(","),
        )
        print(f" | > Loaded eval dataset from {args.data_dir}. Size: {len(eval_dataset)} ...")
    else:
        raise ValueError(" | > Data directory not provided")

    lora_config = LoraConfig(
        r=args.lora_r,
        bias="none",
        inference_mode=False,
        task_type="SEQ_2_SEQ_LM",
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
    )

    model.set_label_smoothing(args.label_smoothing)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f" | > Loading metrics factory with BLEU and chrF ...")
    seq2seq_compute_metrics = compute_metrics_factory(
        tokenizer=tokenizer,
        print_samples=args.print_samples,
        metric_dict={"BLEU": bleu_metric, "chrF": chrf_metric},
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        fp16=True, # use fp16 for faster training
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=100,
        save_total_limit=1,
        predict_with_generate=True,
        load_best_model_at_end=True,
        max_steps=args.max_steps, # max_steps overrides num_train_epochs
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_accumulation_steps=args.grad_accum_steps,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.num_workers,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to=args.report_to,
        generation_max_length=256,
        generation_num_beams=5,
        sortish_sampler=True,
        group_by_length=True,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
        dataloader_prefetch_factor=2,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=seq2seq_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
                early_stopping_threshold=args.threshold,
            )
        ],
    )

    print(f" | > Starting training ...")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print(f" | > Training interrupted ...")

    # this will only save the LoRA adapter weights
    model.save_pretrained(args.output_dir)



if __name__ == "__main__":
    parser = get_arg_parse()
    args = parser.parse_args()

    main(args)
