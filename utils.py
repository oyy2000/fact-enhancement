def qwen_chat_prompt(
    question: str,
    system: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
) -> str:
    """
    Canonical Qwen chat template (string form).
    No dependency on lm-eval arg_0.
    """
    return (
        "<|im_start|>system\n"
        f"{system}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Q: {question}\n"
        "A: Let's think step by step.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
