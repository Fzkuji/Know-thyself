# Qwen2.5-7B-Instruct on TriviaQA

## Experiment Settings

| Setting | Value |
|---------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Dataset | TriviaQA |
| Train Samples | 20,000 |
| Val Samples | 1,000 |
| Label Mode | binary |
| Learning Rate | 1e-5 |
| Epochs | 10 |
| GPUs | 8 |

## Results

| Epoch | Train Judg | Train QA | Train AUROC | Val Judg | Val QA | Val AUROC |
|-------|------------|----------|-------------|----------|--------|-----------|
| 0     | 66.1%      | 64.9%    | 0.6908      | 68.3%    | 67.9%  | 0.6906    |
| 1     | 83.4%      | 65.4%    | 0.8994      | 80.2%    | 68.5%  | 0.8552    |
| 2     | 90.3%      | 64.2%    | 0.9479      | 80.1%    | 68.4%  | 0.8539    |
| 3     | 91.6%      | 63.1%    | 0.9537      | 78.4%    | 67.3%  | 0.8417    |
| 4     | 93.1%      | 63.3%    | 0.9709      | 77.4%    | 66.9%  | 0.8430    |
| 5     | 93.5%      | 62.8%    | 0.9745      | 79.3%    | 66.4%  | 0.8515    |
| 6     | 93.4%      | 62.2%    | 0.9758      | 78.7%    | 66.3%  | 0.8489    |
| 7     | 94.2%      | 62.5%    | 0.9803      | 78.6%    | 66.2%  | 0.8501    |
| 8     | 94.4%      | 61.5%    | 0.9808      | 79.1%    | 65.8%  | 0.8503    |
| 9     | 94.4%      | 61.6%    | 0.9808      | 79.8%    | 65.5%  | 0.8580    |
| 10    | 95.2%      | 62.0%    | 0.9859      | 79.7%    | 65.2%  | 0.8606    |

## Improvement (Epoch 0 → Epoch 10)

| Metric | Change |
|--------|--------|
| Train Judgment | +29.1% |
| Train QA | -2.9% |
| Train AUROC | +0.2951 |
| Val Judgment | +11.4% |
| Val QA | -2.7% |
| Val AUROC | +0.1699 |

## Analysis

- **Judgment accuracy**: Significant improvement on both train (+29.1%) and validation (+11.4%) sets
- **AUROC**: Strong improvement from ~0.69 to ~0.86 on validation, indicating better calibration
- **QA accuracy**: Slight degradation (-2.7% on val), suggesting minimal catastrophic forgetting
- **Overfitting**: Gap between train (95.2%) and val (79.7%) judgment accuracy suggests some overfitting
- **Best val epoch**: Epoch 1 achieves best val judgment (80.2%) with minimal QA degradation
