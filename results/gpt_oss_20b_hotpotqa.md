# GPT-OSS-20B on HotpotQA

## Experiment Settings

| Setting | Value |
|---------|-------|
| Model | openai/gpt-oss-20b |
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
| 0     | 78.7%      | 21.3%    | 0.5019      | 79.6%    | 20.4%  | 0.5024    |
| 1     | 82.5%      | 24.8%    | 0.8832      | 82.1%    | 21.4%  | 0.8674    |
| 2     | 88.0%      | 27.1%    | 0.9498      | 83.5%    | 23.2%  | 0.9046    |
| 3     | 91.7%      | 26.3%    | 0.9662      | 84.4%    | 22.7%  | 0.9062    |
| 4     | 92.0%      | 25.8%    | 0.9730      | 84.9%    | 21.4%  | 0.9128    |
| 5     | 92.6%      | 23.8%    | 0.9768      | 85.6%    | 19.7%  | 0.9161    |
| 6     | 92.9%      | 23.6%    | 0.9781      | 86.0%    | 19.4%  | 0.9184    |
| 7     | 92.5%      | 22.9%    | 0.9759      | 85.8%    | 18.2%  | 0.9160    |
| 8     | 92.5%      | 21.2%    | 0.9772      | 87.0%    | 16.7%  | 0.9215    |
| 9     | 92.6%      | 21.2%    | 0.9767      | 87.3%    | 16.6%  | 0.9272    |
| 10    | 87.3%      | 13.4%    | 0.9522      | 86.0%    | 11.6%  | 0.9115    |

## Improvement (Epoch 0 → Epoch 10)

| Metric | Change |
|--------|--------|
| Train Judgment | +8.6% |
| Train QA | -7.9% |
| Train AUROC | +0.4503 |
| Val Judgment | +6.4% |
| Val QA | -8.9% |
| Val AUROC | +0.4091 |

## Analysis

- **Baseline QA accuracy**: Very low (~21%), even lower than Qwen2.5-7B on HotpotQA (32%), indicating GPT-OSS-20B struggles with multi-hop reasoning
- **Baseline AUROC**: Near random (~0.50), indicating the model has no initial calibration for self-assessment
- **Judgment accuracy**: Improved from 79.6% to 87.3% (best at epoch 9), with +6.4% final improvement
- **AUROC**: Excellent improvement from 0.50 to 0.91 (+0.41), showing dramatic calibration improvement
- **QA degradation**: Significant decline (-8.9% on val), worse than Qwen experiments
- **Best val epoch**: Epoch 9 achieves best val judgment (87.3%) and AUROC (0.9272)
- **Epoch 10 anomaly**: Both judgment and QA dropped significantly, suggesting potential overfitting or training instability

## Comparison with Qwen2.5-7B-Instruct on HotpotQA

| Metric | Qwen2.5-7B | GPT-OSS-20B |
|--------|------------|-------------|
| Baseline QA | 32.3% | 21.3% |
| Baseline AUROC | 0.6688 | 0.5019 |
| Final Val Judgment | 83.0% | 86.0% |
| Final Val AUROC | 0.8962 | 0.9115 |
| Val Judgment Improvement | +14.7% | +6.4% |
| Val AUROC Improvement | +0.23 | +0.41 |

GPT-OSS-20B shows stronger AUROC improvement despite starting from near-random calibration, achieving slightly better final AUROC. However, Qwen2.5-7B has better baseline QA accuracy and less QA degradation during training.
