# NN Pretraining Strategies

Extension of [Finke et al. (2023)](https://arxiv.org/abs/2309.13111) investigating whether pretraining closes the BDT–NN performance gap in weakly supervised anomaly detection on the LHCO R&D dataset.

## Running experiments

### BDT (no pretraining)

```bash
# Baseline
python run_pipeline.py --mode IAD --classifier BDT --input_set baseline \
    --data_file <path_to_data> --extrabkg_file <path_to_extrabkg> \
    --directory results/IAD_baseline_BDT/

# Gaussian noise (N in {1, 2, 5, 10, 30, 50})
python run_pipeline.py --mode IAD --classifier BDT --input_set baseline --gaussian_inputs N \
    --directory results/IAD_BDT_NG/

# Extended feature sets (A in {extended1, extended2, extended3})
python run_pipeline.py --mode IAD --classifier BDT --input_set A \
    --directory results/IAD_BDT_A/
```

### NN (with optional pretraining)

Same flags as BDT with `--classifier NN`. Add `--pretrain_strategy` to enable pretraining:

| Strategy | `--pretrain_strategy` | Config file |
|---|---|---|
| Ref-vs-ref | `ref_vs_ref` | `classifier4.yml` |
| Supervised | `supervised` | `classifier4.yml` |
| Autoencoder (variable bottleneck) | `autoencoder` | `classifier4.yml` |
| Autoencoder (bottleneck=4) | `autoencoder` + `--bottleneck 4` | `classifier4.yml` |
| Masked Feature Prediction | `masked_feature` | `classifier_mfp.yml` |
| SCARF SSL | `scarf_ssl` | `classifier_scarfssl.yml` |

Example:

```bash
python run_pipeline.py --mode IAD --classifier NN --input_set baseline --gaussian_inputs 10 \
    --pretrain_strategy autoencoder \
    --cl_filename classifier4.yml \
    --directory results/IAD_NN_autoencoder_10G/
```

Use `--N_runs` and `--start_at_run` to parallelise across jobs.

## Ensembling NN results

After runs complete, aggregate into ensemble ROC curves:

```bash
python ensemble_NNs.py --directory results/IAD_NN_autoencoder_10G/
```

Writes `fpr/tpr_50_temp.npy` (ensemble of 50) and `fpr/tpr_10_temp.npy` (ensemble of 10) into the results directory.

## Analysis

Open `pretraining_analysis.ipynb`. Set `RES` at the top to point to your results directory. Produces Fig 2 (Gaussian noise robustness) and Fig 3 (extended feature sets) SIC curves for each pretraining strategy, plus cross-method comparisons.
