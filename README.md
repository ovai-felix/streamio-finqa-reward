# streamio-finqa-reward

Reward function for StreamIO GRPO jobs on FinQA-style data.

```json
"reward": {"package": "git+https://github.com/ovai-felix/streamio-finqa-reward@<commit>",
           "entrypoint": "streamio_finqa_reward:reward"}
```

`reward(prompt, completion, target)` returns a tiered score in [0, 1]: 0.1 for a parseable `#### <answer>`,
+0.2 within 5% of the gold answer, +0.7 exact (within 0.5%). No dependencies beyond the standard library.
