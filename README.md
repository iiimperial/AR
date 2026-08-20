# AR
This repository provides **implementation** for paper "Automated Detection of Apical Rocking in Echocardiographic Videos Using Deep Learning".


## Pipeline Overview
Two‑stage pipeline for Apical Rocking (AR) detection from apical‑4‑chamber echocardiogram video:
1. **Stage1 LV Segmentation**: U‑Net + Shape‑Prior‑Module(SPM), segment left‑ventricular endocardium, suppress non‑anatomic background noise.
2. **Stage2 Spatiotemporal Classification**:
   - Primary model: ResNet + LSTM (CNN+LSTM), classify `AR / No‑AR`
   - Comparator model: ResNet + Longformer‑Transformer (CNN+Transformer)

## Environment
Python 3.6
pip install -r requirements.txt
## Pretrained weights
ResNet‑18 weight is available at [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)

## Quick Start
1. Data preprocessing
> python data/preprocess.py --dicom_dir ./raw_dicom --out_dir ./processed --n_sample_frames 16
2. Train segmentation model
> python train/train_segmentation.py --config configs/seg_config.yaml
3. Train classification model
> python train/train_classifier.py --config configs/cls_config.yaml --model cnnlstm
4. Run inference on one echocardiogram video
> python inference/run_inference.py \
    --seg_ckpt ./checkpoints/seg_best.pth \
    --cls_ckpt ./checkpoints/cls_cnnlstm_best.pth \
    --input_dicom ./example_video.dcm \
    --output_json ./result.json
5. Evaluate model performance
### Evaluation (segmentation)
> python evaluate/eval_segmentation.py --pred_dir ./pred_mask --gt_dir ./gt_mask
### Evaluation (classification)
> python evaluate/eval_classification.py --pred_csv ./pred.csv --gt_csv ./label.csv
