# CIMemories: A Compositional Benchmark for Contextual Integrity of Persistent Memory in LLMs


<p align="center">
  <img src="fig-cimemories.png" alt="CIMemories Overview" width="800"/>
</p>

Large Language Models (LLMs) increasingly use persistent memory from past interactions to enhance personalization and task performance. However, this memory introduces **critical risks when sensitive information is revealed in inappropriate contexts**. Knowing what personal information to reveal—and what to keep private—is just as important as remembering it in the first place.

**[CIMemories](https://arxiv.org/abs/2511.14937)** is a benchmark for evaluating whether LLMs appropriately control information flow from memory based on task context. Using synthetic user profiles with over 100 attributes per user, paired with diverse task contexts, CIMemories tests whether each attribute is revealed appropriately—essential for some tasks but inappropriate for others.

In our accompanying paper, we find that frontier models exhibit **up to 69% attribute-level violations** (leaking information inappropriately), with lower violation rates often coming at the cost of task utility. Violations accumulate across both tasks and runs: as usage increases from 1 to 40 tasks, GPT-4o's violations rise from 0.1% to 9.6%, reaching **25.1% when the same prompt is executed 5 times**, revealing arbitrary and unstable behavior in which models leak different attributes for identical prompts. 

**Privacy-conscious prompting does not solve this**—models overgeneralize, sharing everything or nothing rather than making nuanced, context-dependent decisions. These findings reveal fundamental limitations that require contextually aware reasoning capabilities, not just better prompting or scaling.


## CIMemories data on HuggingFace

CIMemories data is now available on 🤗 HuggingFace!

```python
import datasets

cimemories = datasets.load_dataset('https://huggingface.co/datasets/facebook/CIMemories')
```

For the full CIMemories pipeline, including synthetic profile generation, evaluation, and computing metrics, see below.

## Installation

Make sure you have [conda](https://docs.conda.io/en/latest/). Then clone the repo and set up the environment:

```bash
git clone https://github.com/facebookresearch/CIMemories.git
cd CIMemories
conda env create -f env.yaml
```

Before you do anything, enter the environment:

```bash
conda activate cimemories
```

## Evaluating a Model against CIMemories

The raw synthetic user profiles (i.e., the attributes) and the task contexts are included in [this file](data/data_openai_gpt-oss-120b.json). 

### 1. Generating Contextual Privacy Labels

First, you must generate the \{0,1\} labels for attribute-context pairs (whether the attribute should be shared or not in a particular context). We use GPT-5 to generate the labels. To do so, set the following environment variables (AZURE_API_BASE_gpt_5, AZURE_API_KEY_gpt_5, AZURE_API_VERSION_gpt_5) using the endpoint, API key, and API version as is standard when using an azure deployment of any OpenAI model. Then just run:

```
cd data
python persona_gold_labels.py
```

Concretely, this will store an updated version of the data that includes the labels in a new file. Feel free to let this run overnight, it may take a while.

### 2. Running

Launch a DeepSeek-R1 instance for the judge:
```
vllm serve deepseek-ai/DeepSeek-R1-0528 --tensor-parallel-size=8 --reasoning-parser deepseek_r1 --port 8000
```

Now to run an experiment, you'll use exp.sh. Make any modifications here as needed:
- Set the host and port via STRONG_MODEL_HOST and STRONG_MODEL_PORT to be the host and port where deepseek is hosted.
- Also set the model via MODEL_NAME to the one you want to evaluate. For example, you can host a vllm instance of some open source model, and just point to it here again by specifying MODEL_HOST and MODEL_PORT. The code by default also supports some closed source models, e.g., azure deployments of GPT-4o, o3, GPT-5 by just setting the appropriate environment variables specified in the AsyncModel class of eval.py:28-81 (an example is provided as commented out in exp.sh). If you want to evaluate a custom closed source model/provider, it should be extremely easy to add support within a few lines to the AsyncModel class of eval.py:28-81.
- Finally, set number of profiles, number of trials, etc as desired. Then run as:

```
cd eval
./exp.sh
```

This should store a results.jsonl in the results directory. You can compute violation and coverage metrics on it as:
```
python metrics.py --path path_to_results_jsonl
```
Which will print violation and coverage per-profile as well as aggregated across profiles.

## Contributing

Please see [CONTRIBUTING](CONTRIBUTING.md).

## License

Please see the [LICENSE](LICENSE) file.

Please note: Third party content pulled from other locations are subject to its own licenses and you may have other legal obligations or restrictions that govern your use of that content.

## Citation

```bibtex
@misc{mireshghallah2025cimemories,
      title={CIMemories: A Compositional Benchmark for Contextual Integrity of Persistent Memory in LLMs}, 
      author={Mireshghallah, Niloofar and Mangaokar, Neal and Kokhlikyan, Narine and Zharmagambetov, Arman and Zaheer, Manzil and Mahloujifar, Saeed and Chaudhuri, Kamalika},
      year={2025},
      eprint={2511.14937},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2511.14937}, 
}
```

