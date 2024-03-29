import os
import json
import argparse
from sacrebleu.metrics import BLEU, CHRF
from peft import LoraConfig, get_peft_model
from datasets import Dataset, concatenate_datasets
from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    EarlyStoppingCallback,
)

bleu_metric = BLEU()
chrf_metric = CHRF()

def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "ai4bharat/indictrans2-en-indic-1B",
            "ai4bharat/indictrans2-indic-en-1B",
            "ai4bharat/indictrans2-indic-indic-1B",
            "ai4bharat/indictrans2-en-indic-dist-200M",
            "ai4bharat/indictrans2-indic-en-dist-200M",
            "ai4bharat/indictrans2-indic-indic-dist-320M",
        ],
    )
    parser.add_argument(
        "--direction",
        type=str,
        required=True,
        choices=["en-indic", "indic-en", "indic-indic"],
    )
    parser.add_argument(
        "--src_lang_list",
        type=str,
        required=True,
        help="comma separated list of source languages",
    )
    parser.add_argument(
        "--tgt_lang_list",
        type=str,
        required=True,
        help="comma separated list of target languages",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
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
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--mixed_precision", type=str, default="none", choices=["fp16", "bf16", "none"])
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "azure_ml", "none"],
    )
    parser.add_argument("--patience", type=int, default=5),
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument(
        "--generation_config",
        type=json.loads,
        default='{"max_new_tokens": 256, "min_length": 1, "num_beams": 5, "use_cache": true, "num_return_sequences": 1}',
    )
    return parser


def load_and_process_translation_dataset(
    data_dir,
    split="train",
    tokenizer=None,
    processor=None,
    src_lang_list=None,
    tgt_lang_list=None,
):
    complete_dataset = None

    for src_lang in src_lang_list:
        for tgt_lang in tgt_lang_list:
            if src_lang == tgt_lang:
                continue
            src_path = os.path.join(data_dir, split, f"{src_lang}-{tgt_lang}", f"{split}.{src_lang}")
            tgt_path = os.path.join(data_dir, split, f"{src_lang}-{tgt_lang}", f"{split}.{tgt_lang}")
            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                raise FileNotFoundError(
                    f"Source ({split}.{src_lang}) or Target ({split}.{tgt_lang}) file not found in {data_dir}"
                )
            with open(src_path, encoding="utf-8") as src_file, open(tgt_path, encoding="utf-8") as tgt_file:
                src_lines = src_file.readlines()
                tgt_lines = tgt_file.readlines()

            # Ensure both files have the same number of lines
            assert len(src_lines) == len(
                tgt_lines
            ), f"Source and Target files have different number of lines for {split}.{src_lang} and {split}.{tgt_lang}"

            # Pairing each source line with its corresponding target line
            data = {
                "sentence_SRC": [x.strip() for x in src_lines],
                "sentence_TGT": [x.strip() for x in tgt_lines],
            }

            dataset = Dataset.from_dict(data)

            dataset = dataset.map(
                lambda example: preprocess_fn(
                    example,
                    tokenizer=tokenizer,
                    processor=processor,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    max_source_length=tokenizer.model_max_length,
                    max_target_length=tokenizer.model_max_length,
                ),
                batched=True,
            )

            complete_dataset = (
                dataset if complete_dataset is None else concatenate_datasets([complete_dataset, dataset])
            )

    return complete_dataset


def compute_metrics_factory(tokenizer, metric_dict=None):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = tokenizer.batch_decode(preds, src=False)
        labels = tokenizer.batch_decode(labels, src=False)

        return {
            metric_name: metric.compute(predictions=preds, references=[labels]).score
            for (metric_name, metric) in metric_dict.items()
        }

    return compute_metrics


def preprocess_fn(example, tokenizer, processor, src_lang, tgt_lang, **kwargs):
    inputs = processor.preprocess_batch(example["sentence_SRC"], src_lang=src_lang, tgt_lang=tgt_lang, is_target=False)
    targets = processor.preprocess_batch(example["sentence_TGT"], src_lang=tgt_lang, tgt_lang=src_lang, is_target=True)

    model_inputs = tokenizer(
        inputs,
        src=True,
        truncation=True,
        padding="max_length",
        max_length=kwargs["max_source_length"],
    )
    labels = tokenizer(
        targets,
        src=False,
        truncation=True,
        padding="max_length",
        max_length=kwargs["max_target_length"],
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    print(f" | > Loading {args.model_name} and {args.direction} tokenizer ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, trust_remote_code=True, dropout=args.dropout)

    tokenizer = IndicTransTokenizer(direction=args.direction)
    processor = IndicProcessor(inference=False)

    if args.data_dir is not None:
        print(f" | > Loading train dataset from {args.data_dir} ...")
        train_dataset = load_and_process_translation_dataset(
            args.data_dir,
            split="train",
            tokenizer=tokenizer,
            processor=processor,
            src_lang_list=args.src_lang_list.split(","),
            tgt_lang_list=args.tgt_lang_list.split(","),
        )

        print(f" | > Loading dev dataset from {args.data_dir} ...")
        eval_dataset = load_and_process_translation_dataset(
            args.data_dir,
            split="dev",
            tokenizer=tokenizer,
            processor=processor,
            src_lang_list=args.src_lang_list.split(","),
            tgt_lang_list=args.tgt_lang_list.split(","),
        )
    else:
        raise ValueError(" | > Data directory not provided")

    lora_config = LoraConfig(
        r=args.lora_r,
        bias="none",
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
        metric_dict={"BLEU": bleu_metric, "chrF": chrf_metric},
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=100,
        save_total_limit=1,
        predict_with_generate=True,
        load_best_model_at_end=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
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
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        generation_config=GenerationConfig(**args.generation_config),
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=None,
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
    
    # However, if you wish to merge the LoRA adapters with the base model then you should 
    # comment out the following lines below.
    # model.merge_and_unload().save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = get_arg_parse()
    args = parser.parse_args()

    main(args)
