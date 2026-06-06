"""VLN Full-Episode Online GRPO recipe for verl (design: WorldModel report/016).

Full-episode Habitat rollout -> flatten every NaVIDA decision into an independent
training sample -> trajectory-level GRPO. Self-contained; prompt/parse aligned to
the VLN project's eval_vllm_navida.py.
"""
