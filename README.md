# IndicTrans2

[📜 Paper](https://arxiv.org/abs/2305.16307) | [🌐 Website](https://ai4bharat.iitm.ac.in/indic-trans2) | [▶️ Demo](https://models.ai4bharat.org/#/nmt/v2) | [🤗 HF Inference](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_inference)

IndicTrans2 is the first open-source transformer-based multilingual NMT model that supports high-quality translations across all the 22 scheduled Indic languages — including multiple scripts for low-resouce languages like Kashmiri, Manipuri and Sindhi. It adopts script unification wherever feasible to leverage transfer learning by lexical sharing between languages. Overall, the model supports five scripts Perso-Arabic (Kashmiri, Sindhi, Urdu), Ol Chiki (Santali), Meitei (Manipuri), Latin (English), and Devanagari (used for all the remaining languages).

We open-souce all our training dataset (BPCC), back-translation data (BPCC-BT), final IndicTrans2 models, evaluation benchmarks (IN22, which includes IN22-Gen and IN22-Conv) and training and inference scripts for easier use and adoption within the research community. We hope that this will foster even more research in low-resource Indic languages, leading to further improvements in the quality of low-resource translation through contributions from the research community.

This code repository contains instructions for downloading the artifacts associated with IndicTrans2, as well as the code for training/fine-tuning the multilingual NMT models.

Here is the list of languages supported by the IndicTrans2 models:

<table>
<tbody>
  <tr>
    <td>Assamese (asm_Beng)</td>
    <td>Kashmiri (Arabic) (kas_Arab)</td>
    <td>Punjabi (pan_Guru)</td>
  </tr>
  <tr>
    <td>Bengali (ben_Beng)</td>
    <td>Kashmiri (Devanagari) (kas_Deva)</td>
    <td>Sanskrit (san_Deva)</td>
  </tr>
  <tr>
    <td>Bodo (brx_Deva)</td>
    <td>Maithili (mai_Deva)</td>
    <td>Santali (sat_Olck)</td>
  </tr>
  <tr>
    <td>Dogri (doi_Deva)</td>
    <td>Malayalam (mal_Mlym)</td>
    <td>Sindhi (Arabic) (snd_Arab)</td>
  </tr>
  <tr>
    <td>English (eng_Latn)</td>
    <td>Marathi (mar_Deva)</td>
    <td>Sindhi (Devanagari) (snd_Deva)</td>
  </tr>
  <tr>
    <td>Konkani (gom_Deva)</td>
    <td>Manipuri (Bengali) (mni_Beng)</td>
    <td>Tamil (tam_Taml)</td>
  </tr>
  <tr>
    <td>Gujarati (guj_Gujr)</td>
    <td>Manipuri (Meitei) (mni_Mtei)</td>
    <td>Telugu (tel_Telu)</td>
  </tr>
  <tr>
    <td>Hindi (hin_Deva)</td>
    <td>Nepali (npi_Deva)</td>
    <td>Urdu (urd_Arab)</td>
  </tr>
  <tr>
    <td>Kannada (kan_Knda)</td>
    <td>Odia (ory_Orya)</td>
    <td></td>
  </tr>
</tbody>
</table>


## Updates

- 🚨 Sep 9, 2023 - Added HF compatible IndicTrans2 models. Please refer to the [README](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_inference) for detailed example usage.
- 🚨 Dec 1, 2023 - Release of Indic-Indic model and corresponding distilled variants for each base model. Please refer to the [Download section](https://github.com/AI4Bharat/IndicTrans2#multilingual-translation-models) for the checkpoints.

## Tables of Contents

- [Download Models and Other Artifacts](#download-models-and-other-artifacts)
  - [Multilingual Translation Models](#multilingual-translation-models)
  - [Training Data](#training-data)
  - [Evaluation Data](#evaluation-data)
- [Installation](#installation)
- [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Preparing Data for Training](#preparing-data-for-training)
  - [Using our SPM model and Fairseq dictionary](#using-our-spm-model-and-fairseq-dictionary)
  - [Training your own SPM models and learning Fairseq dictionary](#training-your-own-spm-models-and-learning-fairseq-dictionary)
- [Training / Fine-tuning](#training--fine-tuning)
- [Inference](#inference)
  - [Fairseq Inference](#fairseq-inference)
  - [CT2 Inference](#ct2-inference)
- [Evaluations](#evaluations)
  - [Baseline Evaluation](#baseline-evaluation)
- [LICENSE](#license)
- [Citation](#citation)


## Download Models and Other Artifacts


### Multilingual Translation Models

| Model                                     | En-Indic | Indic-En |  Indic-Indic |Evaluations           |
|-------------------------------------------|----------|----------|----------|-----------------------|
| Base (used for benchmarking)          | [download](https://indictrans2-public.objectstore.e2enetworks.net/it2_preprint_ckpts/en-indic-preprint.zip) | [download](https://indictrans2-public.objectstore.e2enetworks.net/it2_preprint_ckpts/indic-en-preprint.zip) | [download](https://indictrans2-public.objectstore.e2enetworks.net/it2_preprint_ckpts/indic-indic.zip) | [translations](https://indictrans2-public.objectstore.e2enetworks.net/translation_outputs.zip) (as of May 10, 2023), [metrics](https://drive.google.com/drive/folders/1lOOdaU0VdRSBgJEsNav5zC7wwLBis9NI?usp=sharing) |
| Distilled          | [download](https://indictrans2-public.objectstore.e2enetworks.net/it2_distilled_ckpts/en-indic.zip) | [download](https://indictrans2-public.objectstore.e2enetworks.net/it2_distilled_ckpts/en-indic.zip) | [download](https://indictrans2-public.objectstore.e2enetworks.net/it2_distilled_ckpts/indic-indic.zip)  |


### Training Data

| Data                                     | URL      |
|------------------------------------------|----------|
| Bharat Parallel Corpus Collection (BPCC) | [download](https://indictrans2-public.objectstore.e2enetworks.net/BPCC.zip) |
| Back-translation (BPCC-BT)               | [download](https://indictrans2-public.objectstore.e2enetworks.net/BT_data.zip) |


### Evaluation Data

| Data                    | URL      |
|-------------------------|----------|
| IN22 test set           | [download](https://indictrans2-public.objectstore.e2enetworks.net/IN22_testset.zip) |
| FLORES-22 Indic dev set | [download](https://indictrans2-public.objectstore.e2enetworks.net/flores-22_dev.zip) |


## Installation

Instructions to setup and install everything before running the code.

```bash
# Clone the github repository and navigate to the project directory.
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2

# Install all the dependencies and requirements associated with the project.
source install.sh
```

Note: We recommend creating a virtual environment with python>=3.7.


## Data


### Training

Bharat Parallel Corpus Collection (BPCC) is a comprehensive and publicly available parallel corpus that includes both existing and new data for all 22 scheduled Indic languages. It is comprised of two parts: BPCC-Mined and BPCC-Human, totaling approximately 230 million bitext pairs. BPCC-Mined contains about 228 million pairs, with nearly 126 million pairs newly added as a part of this work. On the other hand, BPCC-Human consists of 2.2 million gold standard English-Indic pairs, with an additional 644K bitext pairs from English Wikipedia sentences (forming the BPCC-H-Wiki subset) and 139K sentences covering everyday use cases (forming the BPCC-H-Daily subset). It is worth highlighting that BPCC provides the first available datasets for 7 languages and significantly increases the available data for all languages covered.

You can find the contribution from different sources in the following table:

<table>
<tbody>
  <tr>
    <td rowspan="4">BPCC-Mined</th>
    <td rowspan="2">Existing</th>
    <td>Samanantar</th>
    <td>19.4M</th>
  </tr>
  <tr>
    <td>NLLB</th>
    <td>85M</th>
  </tr>
  <tr>
    <td rowspan="2">Newly Added</th>
    <td>Samanantar++</th>
    <td>121.6M</th>
  </tr>
  <tr>
    <td>Comparable</th>
    <td>4.3M</th>
  </tr>
  <tr>
    <td rowspan="5">BPCC-Human</td>
    <td rowspan="3">Existing</td>
    <td>NLLB</td>
    <td>18.5K</td>
  </tr>
  <tr>
    <td>ICLI</td>
    <td>1.3M</td>
  </tr>
  <tr>
    <td>Massive</td>
    <td>115K</td>
  </tr>
  <tr>
    <td rowspan="2">Newly Added</td>
    <td>Wiki</td>
    <td>644K</td>
  </tr>
  <tr>
    <td>Daily</td>
    <td>139K</td>
  </tr>
</tbody>
</table>

Additionally, we provide augmented back-translation data generated by our intermediate IndicTrans2 models for training purposes. Please refer our paper for more details on the selection of sample proportions and sources.

<table>
<tbody>
  <tr>
    <td>English BT data (English Original)</td>
    <td>401.9M</td>
  </tr>
  <tr>
    <td>Indic BT data (Indic Original)</td>
    <td>400.9M</td>
  </tr>
</tbody>
</table>

<br>

### Evaluation

IN22 test set is a newly created comprehensive benchmark for evaluating machine translation performance in multi-domain, n-way parallel contexts across 22 Indic languages. It has been created from three distinct subsets, namely IN22-Wiki, IN22-Web and IN22-Conv. The Wikipedia and Web sources subsets offer diverse content spanning news, entertainment, culture, legal, and India-centric topics.  IN22-Wiki and IN22-Web have been combined and considered for evaluation purposes and released as IN22-Gen. Meanwhile, IN22-Conv the conversation domain subset is designed to assess translation quality in typical day-to-day conversational-style applications.

<table>
<tbody>
  <tr>
    <td>IN22-Gen (IN22-Wiki + IN22-Web)</td>
    <td>1024 sentences</td>
    <td>🤗 <a href="https://huggingface.co/datasets/ai4bharat/IN22-Gen">ai4bharat/IN22-Gen</td>
  </tr>
  <tr>
    <td>IN22-Conv</td>
    <td>1503 sentences</td>
    <td>🤗 <a href="https://huggingface.co/datasets/ai4bharat/IN22-Conv">ai4bharat/IN22-Conv</td>
  </tr>
</tbody>
</table>

You can download the data artifacts released as a part of this work from the [following section](#download-models-and-other-artifacts).


## Preparing Data for Training

BPCC data is organized under different subsets as described above, where each subset contains language pair subdirectories with the sentences pairs. We also provide LaBSE and LASER for the mined subsets of BPCC. In order to replicate our training setup, you will need to combine the data for corresponding language pairs from different subsets and remove overlapping bitext pairs if any.

Here is the expected directory structure of the data:

```bash
BPCC
├── eng_Latn-asm_Beng
│   ├── train.eng_Latn
│   └── train.asm_Beng
├── eng_Latn-ben_Beng
└── ...
```

While we provide deduplicated subsets with the current available benchmarks, we highly recommend performing deduplication using the combined monolingual side of all the benchmarks. You can use the following command for deduplication once you combine the monolingual side of all the benchmarks in the directory.

```python3
python3 scripts/dedup_benchmark.py <in_data_dir> <out_data_dir> <benchmark_dir>
```

- `<in_data_dir>`: path to the directory containing train data for each language pair in the format `{src_lang}-{tgt_lang}`
- `<out_data_dir>`: path to the directory where the deduplicated train data will be written for each language pair in the format `{src_lang}-{tgt_lang}`
- `<benchmark_dir>`: path to the directory containing the language-wise monolingual side of dev/test set, with monolingual files named as `test.{lang}`


### Using our SPM model and Fairseq dictionary

Once you complete the deduplication of the training data with the available benchmarks, you can preprocess and binarize the data for training models. Please download our trained SPM model and learned Fairseq dictionary using the following links for your experiments.

|                     | En-Indic | Indic-En |
|---------------------|----------|----------|
| SPM model | [download](https://indictrans2-public.objectstore.e2enetworks.net/en-indic-spm.zip) | [download](https://indictrans2-public.objectstore.e2enetworks.net/indic-en-spm.zip) |
| Fairseq dictionary  | [download](https://indictrans2-public.objectstore.e2enetworks.net/en-indic-fairseq-dict.zip) | [download](https://indictrans2-public.objectstore.e2enetworks.net/indic-en-fairseq-dict.zip) |

To prepare the data for training En-Indic model, please do the following:
1. Download the SPM model in the experiment directory and rename it as `vocab`.
2. Download the Fairseq dictionary in the experiment directory and rename it as `final_dict`.

Here is the expected directory for training En-Indic model:

```bash
en-indic-exp
├── train
│   ├── eng_Latn-asm_Beng
│   │   ├── train.eng_Latn
│   │   └── train.asm_Beng
│   ├── eng_Latn-ben_Beng
│   └── ...
├── devtest
│   └── all
│       ├── eng_Latn-asm_Beng
│       │   ├── dev.eng_Latn
│       │   └── dev.asm_Beng
│       ├── eng_Latn-ben_Beng
│       └── ...
├── vocab
│   ├── model.SRC
│   ├── model.TGT
│   ├── vocab.SRC
│   └── vocab.TGT
└── final_dict
    ├── dict.SRC.txt
    └── dict.TGT.txt
```

To prepare data for training the Indic-En model, you should reverse the language pair directories within the train and devtest directories. Additionally, make sure to download the corresponding SPM model and Fairseq dictionary and put them in the experiment directory, similar to the procedure mentioned above for En-Indic model training.

You can binarize the data for model training using the following:

```bash
bash prepare_data_joint_finetuning.sh <exp_dir>
```

- `<exp_dir>`: path to the directory containing the raw data for binarization

You will need to follow the same steps for data preparation in case of fine-tuning models.


### Training your own SPM models and learning Fairseq dictionary

If you want to train your own SPM model and learn Fairseq dictionary, then please do the following:

1. Collect a balanced amount of English and Indic monolingual data (we use around 3 million sentences per language-script combination). If some languages have limited data available, increase their representation to achieve a fair distribution of tokens across languages.
2. Perform script unification for Indic languages wherever possible using `scripts/preprocess_translate.py` and concatenate all Indic data into a single file.
3. Train two SPM models, one for English and other for Indic side using the following:
```bash
spm_train --input=train.indic --model_prefix=<model_name> --vocab_size=<vocab_size> --character_coverage=1.0 --model_type=BPE
```
4. Copy the trained SPM models in the experiment directory mentioned earlier and learn the Fairseq dictionary using the following:
```bash
bash prepare_data_joint_training.sh <exp_dir>
```
5. You will need to use the same Fairseq dictionary for any subsequent fine-tuning experiments and refer to the steps described above ([link](#using-our-spm-model-and-fairseq-dictionary)).


## Training / Fine-tuning

After binarizing the data, you can use train.sh to train the models. We provide the default hyperparameters used in this work. You can modify the hyperparameters as per your requirement if needed. If you want to train the model on a customized architecture, then please define the architecture in `model_configs/custom_transformer.py`. You can start the model training with the following command:

```bash
bash train.sh <exp_dir> <model_arch>
```

- `<exp_dir>`: path to the directory containing the binarized data
- `<model_arch>`: custom transformer architecture used for model training

For fine-tuning, the initial steps remain the same. However, the `finetune.sh` script includes an additional argument, `pretrained_ckpt`, which specifies the model checkpoint to be loaded for further fine-tuning. You can perform fine-tuning using the following command:

```bash
bash finetune.sh <exp_dir> <model_arch> <pretrained_ckpt>
```

- `<exp_dir>`: path to the directory containing the binarized data
- `<model_arch>`: custom transformer architecture used for model training
- `<pretrained_ckpt>`: path to the fairseq model checkpoint to be loaded for further fine-tuning

You can download the model artifacts released as a part of this work from the [following section](#download-models-and-other-artifacts).

The pretrained checkpoints have 3 directories, a fairseq model directory and 2 CT-ported model directories. Please note that the CT2 models are provided only for efficient inference. For fine-tuning purposes you should use the `fairseq_model`. Post that you can use the [fairseq-ct2-converter](https://opennmt.net/CTranslate2/guides/fairseq.html) to port your fine-tuned checkpoints to CT2 for faster inference.


## Inference


### Fairseq Inference

In order to run inference on our pretrained models using bash interface, please use the following:

```bash
bash joint_translate.sh <infname> <outfname> <src_lang> <tgt_lang> <ckpt_dir>
```

- `infname`: path to the input file containing sentences
- `outfname`: path to the output file where the translations should be stored
- `src_lang`: source language
- `tgt_lang`: target language
- `ckpt_dir`: path to the fairseq model checkpoint directory

If you want to run the inference using python interface then please execute the following block of code from the root directory:

```python3
from inference.engine import Model

model = Model(ckpt_dir, model_type="fairseq")

sents = [sent1, sent2,...]

# for a batch of sentences
model.batch_translate(sents, src_lang, tgt_lang)

# for a paragraph
model.translate_paragraph(text, src_lang, tgt_lang)
```

### CT2 Inference

In order to run inference on CT2-ported model using python inference then please execute the following block of code from the root directory:

```python3
from inference.engine import Model

model = Model(ckpt_dir, model_type="ctranslate2")

sents = [sent1, sent2,...]

# for a batch of sentences
model.batch_translate(sents, src_lang, tgt_lang)

# for a paragraph
model.translate_paragraph(text, src_lang, tgt_lang)
```


## Evaluations

We consider the chrF++ as our primary metric. Additionally, we also report the BLEU and Comet scores.
We also perform statistical significance tests for each metric to ascertain whether the differences are statistically significant.

In order to run our evaluation scripts, you will need to organize the evaluation test sets into the following directory structure:

```bash
eval_benchmarks
├── flores
│   └── eng_Latn-asm_Beng
│       ├── test.eng_Latn
│       └── test.asm_Beng
├── in22-gen
├── in22-conv
├── ntrex
└── ...
```

To compute the BLEU and chrF++ scores for prediction file, you can use the following command:

```bash
bash compute_metrics.sh <pred_fname> <ref_fname> <tgt_lang>
```

- `pred_fname`: path to the model translations
- `ref_fname`: path to the reference translations
- `tgt_lang`: target language

In order to automate the inference over the individual test sets for En-Indic, you can use the following command:

```bash
bash eval.sh <devtest_data_dir> <ckpt_dir> <system>
```

- `<devtest_data_dir>`: path to the evaluation set with language pair subdirectories (for example, flores directory in the above tree structure)
- `<ckpt_dir>`: path to the fairseq model checkpoint directory
- `<system>`: system name suffix to store the predictions in the format `test.{lang}.pred.{system}`

In case of Indic-En evaluation, please use the following command:

```bash
bash eval_rev.sh  <devtest_data_dir> <ckpt_dir> <system>
```

- `<devtest_data_dir>`: path to the evaluation set with language pair subdirectories (for example, flores directory in the above tree structure)
- `<ckpt_dir>`: path to the fairseq model checkpoint directory
- `<system>`: system name suffix to store the predictions in the format `test.{lang}.pred.{system}`

**_Note: You don’t need to reverse the test set directions for each language pair._**

In case of Indic-Indic evaluation, please use the following command:

```bash
bash pivot_eval.sh <devtest_data_dir> <pivot_lang> <src2pivot_ckpt_dir> <pivot2tgt_ckpt_dir> <system>
```

- `<devtest_data_dir>`: path to the evaluation set with language pair subdirectories (for example, flores directory in the above tree structure)
- `<pivot_lang>`: pivot language (default should be `eng_Latn`)
- `<src2pivot_ckpt_dir>`: path to the fairseq Indic-En model checkpoint directory
- `<pivot2tgt_ckpt_dir>`: path to the fairseq En-Indic model checkpoint directory
- `<system>`: system name suffix to store the predictions in the format test.{lang}.pred.{system}

In order to perform significance testing for BLEU and chrF++ metrics after you have the predictions for different systems, you can use the following command:

```bash
bash compute_comet_metrics_significance.sh <devtest_data_dir>
```

- `<devtest_data_dir>`: path to the evaluation set with language pair subdirectories (for example, flores directory in the above tree structure)

Similarly, to compute the COMET scores and perform significance testing on predictions of different systems, you can use the following	command.

```bash
bash compute_comet_score.sh <devtest_data_dir>
```

- `<devtest_data_dir>`: path to the evaluation set with language pair subdirectories (for example, flores directory in the above tree structure)

Please note that as we compute significance tests with the same script and automate everything, it is best to have all the predictions for all the systems in place to avoid repeating anything. 
Also, we define the systems in the script itself, if you want to try out other systems, make sure to edit it there itself.


### Baseline Evaluation

To generate the translation results for baseline models such as M2M-100, MBART, Azure, Google, and NLLB MoE, you can check the scripts provided in the "baseline_eval" directory of this repository. For NLLB distilled, you can either modify NLLB_MoE eval or use this [repository](https://github.com/pluiez/NLLB-inference). Similarly, for IndicTrans inference, please refer to this [repository](https://github.com/ai4bharat/IndicTrans).

You can download the translation outputs released as a part of this work from the [following section](#download-models-and-other-artifacts).


## LICENSE

The following table lists the licenses associated with the different artifacts released as a part of this work:

| Artifact                                              | LICENSE   |
|-------------------------------------------------------|-----------|
| Existing Mined Corpora (NLLB & Samanantar)            | [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/)       |
| Existing Seed Corpora (NLLB-Seed, ILCI, MASSIVE)           | [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/)       |
| Newly Added Mined Corpora (Samanantar++ & Comparable) | [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/)       |
| Newly Added Seed Corpora (BPCC-H-Wiki & BPCC-H-Daily)               | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |
| Newly Created IN-22 test set (IN22-Gen & IN22-Conv)                          | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |
| Back-translation data (BPCC-BT)                       | [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/)       |
| Model checkpoints                                     | [MIT](https://github.com/ai4bharat/IndicTrans2/blob/main/LICENSE)       |

The mined corpora collection (BPCC-Mined), existing seed corpora (NLLB-Seed, ILCI, MASSIVE), Backtranslation data (BPCC-BT), are released under the following licensing scheme:
* We do not own any of the text from which this data has been extracted.
* We license the actual packaging of this data under the Creative Commons [CC0 license (“no rights reserved”)](https://creativecommons.org/share-your-work/public-domain/cc0/).
* To the extent possible under law, [AI4Bharat](https://ai4bharat.iitm.ac.in/) has waived all copyright and related or neighboring rights to BPCC-Mined, existing seed corpora (NLLB-Seed, ILCI, MASSIVE) and BPCC-BT.

## Citation

```bash
@article{gala2023indictrans2,
  title   = {IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
  author  = {Jay Gala and Pranjal A. Chitale and Raghavan AK and Varun Gumma Sumanth Doddapaneni and and Aswanth Kumar and Janki Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M. Khapra and Raj Dabre and Anoop Kunchukuttan},
  year    = {2023},
  journal = {Transactions on Machine Learning Research},
  url     = {https://openreview.net/forum?id=vfT4YuzAYA}
}
```
