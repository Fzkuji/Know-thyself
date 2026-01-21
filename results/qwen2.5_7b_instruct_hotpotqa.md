# Qwen2.5-7B-Instruct on HotpotQA

## Experiment Settings

| Setting | Value |
|---------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Dataset | HotpotQA |
| Train Samples | 20,000 |
| Val Samples | 1,000 |
| Label Mode | binary |
| Learning Rate | 4e-6 |
| Epochs | 10 |
| GPUs | 8 |

## Results

| Epoch | Train Judg | Train QA | Train AUROC | Val Judg | Val QA | Val AUROC |
|-------|------------|----------|-------------|----------|--------|-----------|
| 0     | 65.7%      | 32.3%    | 0.6771      | 68.3%    | 27.2%  | 0.6688    |
| 1     | 83.1%      | 33.5%    | 0.9084      | 83.1%    | 28.1%  | 0.8836    |
| 2     | 90.2%      | 32.9%    | 0.9541      | 83.0%    | 28.4%  | 0.8933    |
| 3     | 92.5%      | 31.9%    | 0.9681      | 82.8%    | 27.5%  | 0.8949    |
| 4     | 93.6%      | 31.6%    | 0.9774      | 82.8%    | 27.3%  | 0.8977    |
| 5     | 94.2%      | 31.4%    | 0.9789      | 83.2%    | 26.3%  | 0.8955    |
| 6     | 94.5%      | 31.0%    | 0.9831      | 83.0%    | 26.4%  | 0.8923    |
| 7     | 95.1%      | 30.8%    | 0.9866      | 83.4%    | 25.8%  | 0.8952    |
| 8     | 94.9%      | 29.8%    | 0.9842      | 84.6%    | 24.4%  | 0.9030    |
| 9     | 94.9%      | 29.3%    | 0.9849      | 83.4%    | 24.1%  | 0.9006    |
| 10    | 94.7%      | 28.3%    | 0.9852      | 83.0%    | 23.1%  | 0.8962    |

## Improvement (Epoch 0 → Epoch 10)

| Metric | Change |
|--------|--------|
| Train Judgment | +28.9% |
| Train QA | -3.9% |
| Train AUROC | +0.3082 |
| Val Judgment | +14.7% |
| Val QA | -4.1% |
| Val AUROC | +0.2274 |

## Analysis

- **Baseline QA accuracy**: Much lower than TriviaQA (~32% vs ~65%), reflecting HotpotQA's difficulty (multi-hop reasoning)
- **Judgment accuracy**: Strong improvement from 68.3% to 83.0% on validation (+14.7%)
- **AUROC**: Excellent improvement from 0.67 to 0.90 on validation (+0.23), indicating well-calibrated confidence
- **QA degradation**: Moderate decline (-4.1% on val), slightly worse than TriviaQA experiment
- **Overfitting**: Less severe gap (train 94.7% vs val 83.0%) compared to TriviaQA, possibly due to harder task
- **Best val epoch**: Epoch 8 achieves best val judgment (84.6%) and AUROC (0.9030)

## Comparison with TriviaQA

| Metric | TriviaQA | HotpotQA |
|--------|----------|----------|
| Baseline QA | 64.9% | 32.3% |
| Final Val Judgment | 79.7% | 83.0% |
| Final Val AUROC | 0.8606 | 0.8962 |
| Val Judgment Improvement | +11.4% | +14.7% |
| Val AUROC Improvement | +0.17 | +0.23 |

HotpotQA shows stronger judgment learning despite lower baseline QA accuracy, suggesting the model can learn metacognition effectively even on harder tasks.
