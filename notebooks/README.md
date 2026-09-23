# notebooks

## `lora_judge_colab.ipynb` - a self-hosted LoRA judge

Trains a small LoRA adapter (on `distilroberta-base`) that scores whether an
agent's output is *faithful* to the input, on a **free Colab T4 GPU**, then loads
it **CPU-side** for inference. The result is a judge that runs inside agenteval
(or the Streamlit app) with no GPU and no per-call API cost - a self-hosted
alternative to the hosted LLM-as-judge.

Open in Colab, set **Runtime -> T4 GPU**, and run top to bottom. The final cell
shows the `LoRAFaithfulnessJudge` that plugs into the same `Judge` interface as
the deterministic and LLM judges, so its scores flow through the scorecard and
regression gate unchanged. Grow the training set from real agenteval traces to
sharpen it.
