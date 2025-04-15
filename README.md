# Assignment-3-CAP5516
# Nuclei Instance Segmentation with MobileSAM and LoRA

## Dataset
- Download NuInsSeg from https://zenodo.org/records/10518968
- Place images in `data/images` and masks in `data/masks`

## Usage
1. Download MobileSAM checkpoint (`mobile_sam.pt`) from https://github.com/Chaoning/hang/MobileSAM
2. Run training and evaluation: `python train_lora_sam.py`
3. Results saved in `metrics/` (CSV files) and `visuals/` (images)

## Files
- `train_lora_sam.py`: training/evaluation
- `dataset.py`: dataset loading
- `metrics.py`: Dice, AJI, PQ metrics
- `lora_utils.py`: LoRA implementation

## Notes
- Trained on Mac M1 (MPS backend), which limited training speed.
- Metrics are low due to simple segmentation head and 10 epochs.
