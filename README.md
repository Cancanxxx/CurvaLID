# CurvaLID
Github Repository for "Geometric Analysis for Adversarial Prompt Classification Using Curvature and Local Intrinsic Dimension"

## Abstract

Adversarial prompts that can jailbreak large language models (LLMs) and lead to undesirable behaviours pose a significant challenge to the safe deployment of LLMs. Existing defenses, such as input perturbation and adversarial training, depend on activating LLMs' defense mechanisms or fine-tuning LLMs individually, resulting in inconsistent performance across different prompts and LLMs. To address this, we propose CurvaLID, an algorithm that classifies benign and adversarial prompts by leveraging two complementary geometric measures: Local Intrinsic Dimensionality (LID) and curvature. 
LID provides an analysis of geometric differences at the prompt level, while curvature captures the degree of curvature in the manifolds and the semantic shifts at the word level. Together, these tools capture both prompt-level and word-level geometric properties, enhancing adversarial prompt detection.
We demonstrate the limitations of using token-level LID, as applied in previous work, for capturing the geometric properties of text prompts. To address this, we propose PromptLID to calculate LID in prompt-level representations to explore the adversarial local subspace for detection. Additionally, we propose TextCurv to further analyze the local geometric structure of prompt manifolds by calculating the curvature in text prompts. CurvaLID achieves over 0.99 detection accuracy, effectively reducing the attack success rate of advanced adversarial prompts to zero or nearly zero. Importantly, CurvaLID provides a unified detection framework across different adversarial prompts and LLMs, as it achieves consistent performance regardless of the specific LLM targeted.

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
