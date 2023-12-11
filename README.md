# DISCLAIMER !!

This branch of IndicTrans2 specializes in Knowledge Distillation and building compact models, and requires different versions of [fairseq](https://github.com/VarunGumma/fairseq) and [indic_nlp_library](https://github.com/VarunGumma/indic_nlp_library). _Please follow the instructions given in this README only if you want to perform distillation_. For all other purposes, please use the `main` branch which offers a broader range of functionalities and support. This repository also has some scripts optimized and parallelized to process large amounts of data faster.


# Installation
Instructions to setup and install everything before running the code.

```
# Clone the github repository and navigate to the project directory.
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2
git checkout Distillation

# Install all the dependencies and requirements associated with the project.
source install.sh
```

Note: We recommend creating a virtual environment with python>=3.11.

# Data
To distill IT2, we use Word-Level distillation, i.e. online distillation, in which the student tries to mimic the probability distribution of the teacher at each timestep. Hence, we need the same data used for training the original IT2 model. If it is already available in a binarized fashion, skip to next the section, else run the follwing command:

```
# exp_dir - The experiment directory, where the final binarized data and vocab files will be present. Initially, the script will delete this directory if is exsists, so make sure to move out any files you may have previously generated. 

# vocab_dir - Since the KL-Divergence between the student and teacher models is computed as a loss, distilled models need to have the same vocabulary as the teacher. This variable should have the path to the original IT2 vocabulary and dictionaries. You can download them from the links provided in the main README.

# train_data_dir - Path to the training data. Please refer to the `Preparing Data for Training` section in the main README for the structure and arrangement of the language-wise directories and training files. 

# devtest_data_dir - Path to the development data and it should be organized similar to the training data.

# benchmarks_dir - Path to monolingual side of all benchmarks for deduplication.

bash prepare_data.sh $exp_dir $vocab_dir $train_data_dir $devtest_data_dir $benchmarks_dir
```


# Distillation
Once the training data is binarized and ready, download the original IT2 fairseq checkpoints. Run the following command which normalizes the model name and architecture name in the fairseq checkpoint. This negates the need of a special `user_dir` during training, and the [pre-defined transformer architecture variations](https://github.com/VarunGumma/fairseq/blob/main/fairseq/models/transformer/transformer_legacy.py) from [fairseq](https://github.com/VarunGumma/fairseq) can be used. For the distilled models in our paper, we use the `base18L` variant from [Gumma et al.](https://aclanthology.org/2023.eamt-1.11/) 

```
# ckpt_dir - Path to downloaded checkpoint directory

python3 normalize_checkpoints.py $ckpt_dir
```

Run the following command to start the distillation training. Note that, both the student and large teacher model will be loaded into the memory and batch will be forward passed through both models. However, the gradients and updates will be computed and applied to the student model only. It is recommended to use the `--memory-efficient-fp16` flag instead of `--fp16` in this case, as it will be helpful in this compute intensive endevour, and it is recommended to run this training in a distributed setup with atleast 4 GPUs. 

```
# data_dir - Path to binarized training data

# teacher_ckpt_path - Path to teacher model checkpoint

# wandb_project - Name of wandb project for logging and easy visualization [Optional]

bash distill.sh $data_dir $wandb_project 
```


# Finetuning
We finetune the models trained with high-quality seed data. To prepare and binarize the seed data, follow the data preparation process mentioned above and run the following command:

```
# data_dir - Path to binarized fine-tuning data

# restore_from_dir - Path to directory of initially trained model. `checkpoint_best.pt` will be restored, and the finetuning process will continue from there

# wandb_project - Name of wandb project for logging and easy visualization [Optional]

bash finetune.sh $data_dir $restore_from_dir $wandb_project
```

# Evaluation
To evaluate the distilled/finetuned models, run the following command:

```
# devtest_dir - Path to test files, structured similar to the train data

# ckpt_dir - Path to Directory which has model checkpoints. This directory should have the vocab files and may have more than one model checkpoints, including checkpoint of teacher model. 

# model - Model directory in ckpt_dir. Therefore, the checkpoint will be read from ckpt_dir/model

# system - System name to the added to the prediction files to track them later [Optional]

bash eval.sh $devtest_dir $ckpt_dir $model $system
```

# Artifacts
We release the distilled models trained as part of this work under the same license and the links to download the fairseq models are available in the `main` README. The distilled models are also hosted on HuggingFace: [Indic-En](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M), [En-Indic](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M). The tokenizer for the HF models is available on the `main` branch [here](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_inference).


# Blog Post
We also release a [blog](https://ai4bharat.iitm.ac.in/indictrans2-m2m/) with some additional details and statistics about the distilled models. 


# Citation
Please cite the following two papers if you use our distilled models/codebase/procedure, or [fairseq](https://github.com/VarunGumma/fairseq) clone:

```
@article{gala2023indictrans2,
  title   = {IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
  author  = {Jay Gala and Pranjal A. Chitale and Raghavan AK and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar and Janki Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M. Khapra and Raj Dabre and Anoop Kunchukuttan},
  year    = {2023},
  journal = {Transactions on Machine Learning Research},
  url     = {https://openreview.net/forum?id=vfT4YuzAYA}
}
```
```
@inproceedings{gumma-etal-2023-empirical,
    title = "An Empirical Study of Leveraging Knowledge Distillation for Compressing Multilingual Neural Machine Translation Models",
    author = "Gumma, Varun  and
      Dabre, Raj  and
      Kumar, Pratyush",
    editor = "Nurminen, Mary  and
      Brenner, Judith  and
      Koponen, Maarit  and
      Latomaa, Sirkku  and
      Mikhailov, Mikhail  and
      Schierl, Frederike  and
      Ranasinghe, Tharindu  and
      Vanmassenhove, Eva  and
      Vidal, Sergi Alvarez  and
      Aranberri, Nora  and
      Nunziatini, Mara  and
      Escart{\'\i}n, Carla Parra  and
      Forcada, Mikel  and
      Popovic, Maja  and
      Scarton, Carolina  and
      Moniz, Helena",
    booktitle = "Proceedings of the 24th Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2023",
    address = "Tampere, Finland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2023.eamt-1.11",
    pages = "103--114",
    abstract = "Knowledge distillation (KD) is a well-known method for compressing neural models. However, works focusing on distilling knowledge from large multilingual neural machine translation (MNMT) models into smaller ones are practically nonexistent, despite the popularity and superiority of MNMT. This paper bridges this gap by presenting an empirical investigation of knowledge distillation for compressing MNMT models. We take Indic to English translation as a case study and demonstrate that commonly used language-agnostic and language-aware KD approaches yield models that are 4-5x smaller but also suffer from performance drops of up to 3.5 BLEU. To mitigate this, we then experiment with design considerations such as shallower versus deeper models, heavy parameter sharing, multistage training, and adapters. We observe that deeper compact models tend to be as good as shallower non-compact ones and that fine-tuning a distilled model on a high-quality subset slightly boosts translation quality. Overall, we conclude that compressing MNMT models via KD is challenging, indicating immense scope for further research.",
}
```
