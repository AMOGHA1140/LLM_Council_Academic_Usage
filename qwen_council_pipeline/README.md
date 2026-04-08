1) `debate_output_moe_qwen_gemma.jsonl` - Critic is qwen3.5-35B-A4B, Proponent is gemma4-26B-A4B, Chair is gemma4-26B-A4B. Evaluated on the full 30 papers. 16/30 accuracy.

2) `sample_debate.jsonl` - Evaluated on just 10 papers. 6/10 accuracy. Proponent and chair are  gemma4-26B-A4B, critic is qwen3.5-27B (not MOE)

3) `new_debate_output_moe_qwen_gemma.jsonl` - Evaluated on full 30 papers, proponent and chair are gemma4-26B-A4B, qwen3.5-35B-A4B, similar to (1) but with a different prompt - rejecting paper by default, and accepting only if critic fails. 

4) `new_debate_output_moe_qwen_gemma_rounds=3.jsonl` - same as (3) with 3 rounds instead of 2