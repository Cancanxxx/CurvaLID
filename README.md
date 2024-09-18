# CurvaLID
Github Repository for "Geometric Analysis for Adversarial Prompt Classification Using Curvature and Local Intrinsic Dimension"

## Abstract

Adversarial prompts that can jailbreak large language models (LLMs) and lead to undesirable behaviors pose a significant challenge to their safe deployment. Existing defenses, such as input perturbation and adversarial training, depend on activating LLMs' defense mechanisms or fine-tuning LLMs individually, resulting in inconsistent performance across different prompts and LLMs. To address this, we propose CurvaLID, an algorithm that classifies benign and adversarial prompts by leveraging the Local Intrinsic Dimensionality (LID) and curvature. We demonstrate the limitations of trivially applying LID estimation at the word level for capturing the geometric properties of text prompts. To address this, we use prompt-level representations to explore the adversarial local subspace for detection. Additionally, to further analyze the local geometric structure of prompt manifolds, we propose TextCurv for calculating the curvature in text prompts. CurvaLID achieves over 99\% classification accuracy, effectively reducing the attack success rate of several advanced adversarial prompts to zero or nearly zero. Importantly, CurvaLID provides a unified defense across different adversarial prompts and LLMs, as its performance remains entirely independent of the specific LLM being targeted.

## Code

This is the code for calculating TextCurv
```python
def TextCurv(embeddings):
    curvatures = []
    for i in range(1, len(embeddings)):
        p0 = embeddings[i - 1]
        p1 = embeddings[i]
        norm_p0 = np.linalg.norm(p0)
        norm_p1 = np.linalg.norm(p1)

        cosine_angle = np.dot(p0, p1) / (norm_p0 * norm_p1)
        angular_change = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        distance_change = 1/norm_p0 + 1/norm_p1
        curvature = angular_change / distance_change
        curvatures.append(curvature)
    return curvatures
```
The demo code for CurvaLID can be found at demo.ipynb
