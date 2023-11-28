# IndicTrans2 HF Compatible Models

In this section, we provide details on how to use our [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) models which were originally trained with the [fairseq](https://github.com/facebookresearch/fairseq) to [HuggingFace transformers](https://huggingface.co/docs/transformers/index) for inference purpose. Our scripts for HuggingFace compatible models are adapted from [M2M100 repository](https://github.com/huggingface/transformers/tree/main/src/transformers/models/m2m_100).


### Setup

To get started, follow these steps to set up the environment:

```
# Clone the github repository and navigate to the project directory.
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2

# Install all the dependencies and requirements associated with the project for running HF compatible models.
source install.sh
```

> Note: The `install.sh` script in this directory is specifically for running HF compatible models for inference.


### Converting

In order to convert the fairseq checkpoint to a PyTorch checkpoint that is compatible with HuggingFace Transformers, use the following command:

```bash
python3 convert_indictrans_checkpoint_to_pytorch.py --fairseq_path <fairseq_checkpoint_best.pt> --pytorch_dump_folder_path <hf_output_dir>
```
- `<fairseq_checkpoint_best.pt>`: path to the fairseq `checkpoint_best.pt` that needs to be converted to HF compatible models
- `<hf_output_dir>`: path to the output directory where the HF compatible models will be saved


### Models

| Model    | 🤗 HuggingFace Checkpoints        |
|----------|-----------------------------------|
| En-Indic | [ai4bharat/indictrans2-en-indic-1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) |
| Indic-En | [ai4bharat/indictrans2-indic-en-1B](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B) |
| Distilled En-Indic | [ai4bharat/indictrans2-en-indic-dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) |
| Distilled Indic-En | [ai4bharat/indictrans2-indic-en-dist-200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) |
| Indic-Indic (Stitched) | [ai4bharat/indictrans2-indic-indic-1B](https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B) |
| Distilled Indic-Indic (Stitched) | [ai4bharat/indictrans2-indic-indic-dist-320M](https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M) |


### Inference

With the conversion complete, you can now perform inference using the HuggingFace Transformers. 

You can start with the provided `example.py` script and customize it for your specific translation use case:

```bash
python3 example.py
```

Feel free to modify the `example.py` script to suit your translation needs.

### Citation

```
@article{ai4bharat2023indictrans2,
  title   = {IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
  author  = {AI4Bharat and Jay Gala and Pranjal A. Chitale and Raghavan AK and Sumanth Doddapaneni and Varun Gumma and Aswanth Kumar and Janki Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M. Khapra and Raj Dabre and Anoop Kunchukuttan},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2305.16307}
}
```
