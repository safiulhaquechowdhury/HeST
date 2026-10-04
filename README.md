# HeST: Hesitation-aware Soft Targets

Code, executed experiment notebook and results for the paper

**HeST: A hesitation-aware soft-target framework for transferring human reaction-time knowledge to deep image classifiers**
Safiul Haque Chowdhury, Rakib Ahammed Diptho, Pial Ghosh

HeST converts annotator reaction time from CIFAR-10H into image-specific soft training targets for deep image
classifiers (ResNet-18, WRN-16-4) and compares them with soft labels and two matched controls
(constant softening and within-class shuffled hesitation).

## Contents

| Path | Description |
|---|---|
| `HeST_experiments.ipynb` | The fully executed experiment notebook, with all outputs (backbones, ε selection, main study, tables) |
| `src/human_rt.py` | Cleans the CIFAR-10H trials and computes the annotator-normalised hesitation score of every image |
| `src/targets.py` | Soft labels, constant softening, shuffled hesitation, HeST and the ablation variants |
| `src/models.py` | CIFAR ResNet-18 and WRN-16-4 with an optional hesitation head |
| `src/train_backbone.py` | Backbone pretraining on the CIFAR-10 training set |
| `src/finetune_human.py` | 5-fold cross-validated fine-tuning on CIFAR-10H, temperature scaling, clean/shift/OOD evaluation |
| `src/metrics.py` | Accuracy, human NLL, ECE, Brier, AURC, hesitation alignment, OOD AUROC |
| `src/figures_*.py`, `src/figstyle.py`, `src/data.py` | Figures, tables and data loading |
| `results/human_image_table.csv` | Per-image hesitation scores, vote distributions and reliability halves for all 10,000 images |
| `results/<study>/results.jsonl` | One row per method × seed × fold with every metric |
| `results/chosen_eps.json` | Softening strength selected on fold 0 |
| `results/logs/final_run_log.txt` | Log of the final run, including per-run wall-clock times |

## Data

The datasets are not redistributed here. Download them into `data/`:

- CIFAR-10H: https://github.com/jcpeterson/cifar-10h (place in `data/cifar-10h`)
- CIFAR-10 and CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- SVHN: http://ufldl.stanford.edu/housenumbers/

CIFAR-10, CIFAR-100 and SVHN are downloaded automatically by torchvision on the first run.

## Running

1. Install PyTorch with CUDA for your GPU, then `pip install -r requirements.txt`.
2. Open `HeST_experiments.ipynb` and run all cells. The run resumes automatically after interruption.

The reported results were produced on a single NVIDIA GTX 1070 (8 GB) with three seeds and folds 1–4 of a
stratified 5-fold split; fold 0 was used only to select the softening strength.

## License

Code is released under the MIT License. The datasets keep their original licenses.
