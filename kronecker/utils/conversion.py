#!/usr/bin/env python
import math
import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from kronecker.layers.kronecker_linear import KroneckerLinear
#!/usr/bin/env python
import math
import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
from kronecker.layers.kronecker_linear import KroneckerLinear
import copy


def _get_parent_and_attr(model: nn.Module, full_name: str):
    """
    Given e.g. 'encoder.layer.3.attention.self.query',
    returns (parent_module, 'query'), handling numeric ModuleList indices.
    """

    
    *path, attr = full_name.split(".")
    parent = model
    for p in path:
        if p.isdigit() and isinstance(parent, (list, tuple, nn.ModuleList)):
            parent = parent[int(p)]
        elif hasattr(parent, p):
            parent = getattr(parent, p)
        else:
            raise AttributeError(f"Can't traverse '{p}' in '{full_name}'")
    return parent, attr


def divisors(x: int):
    for i in range(1, int(math.isqrt(x)) + 1):
        if x % i == 0:
            yield i
            if i != x // i:
                yield x // i


def compute_kron_factors(
    in_features: int,
    out_features: int,
    compression: float,
    *,
    prefer_smaller_b: bool = False
):
    """
    Enumerate all (m1,m2),(n1,n2) with
      m1*m2 == out_features, n1*n2 == in_features,
    keep those with orig_params/kron_params >= compression,
    then pick the one whose actual ratio is closest to target,
    breaking ties by smaller B if requested.
    """
    orig = in_features * out_features
    candidates = []

    for m2 in divisors(out_features):
        m1 = out_features // m2
        if m2 > m1:
            continue
        for n2 in divisors(in_features):
            n1 = in_features // n2
            if n2 > n1:
                continue

            kron_params = m1 * n1 + m2 * n2
            ratio = orig / kron_params
            if ratio >= compression:
                candidates.append((ratio, m2 * n2, (m1, m2), (n1, n2)))

    if not candidates:
        raise ValueError(f"No valid Kron factors for compression={compression}")

    # sort by ratio (closest to target from above), then by B-size if needed
    key_fn = (lambda x: (x[0], x[1])) if prefer_smaller_b else (lambda x: x[0])
    _, _, out_fac, in_fac = min(candidates, key=key_fn)
    return out_fac, in_fac


def replace_linears_with_kron(
    model: PreTrainedModel,
    compression: dict = None,
    *,
    prefer_smaller_b: bool = True
) -> PreTrainedModel :
    """
    1) Collect all nn.Linear modules.
    2) Decide which compression factor to use (you can replace *all* linears if you like).
    3) Compute kron factors, build the KroneckerLinear, and swap it in.
    """


    # 1) collect
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            to_replace.append((name, module))
    # 2) replace
    for name, linear in to_replace:
        lname = name.lower()
        # pick a factor; here we just show how you'd pick:
        if 'attn' in lname or 'attention' in lname:
            C = compression['attention']
        elif 'mlp' in lname or 'ffn' in lname or 'intermediate' in lname:
            C = compression['ffn']
        elif 'lm_head' in lname or 'head' in lname:
            C = compression['head']
        else:
            # skip or default
            C = compression.get('ffn')
        if C:
            (m1, m2), (n1, n2) = compute_kron_factors(
                linear.in_features,
                linear.out_features,
                C,
                prefer_smaller_b=prefer_smaller_b
            )
            print(f"[{name}] → out=({m1},{m2}), in=({n1},{n2}), C={C:.1f}")

            kron = KroneckerLinear.from_linear_with_factors(
                linear,
                out_factors=(m1, m2),
                in_factors=(n1, n2),
                num_sum=1,
                efficient_sum=True
            )
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, kron)
    
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--attn_C", type=float, default=4)
    parser.add_argument("--ffn_C", type=float, default=4)
    parser.add_argument("--head_C", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model…")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Orig params: {orig_params:,}")
    # Test text generation
    print("\nGenerating text...")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")


    model = replace_linears_with_kron(
        model,
        compression={
            'attention': args.attn_C,
            'ffn': args.ffn_C,
            'head': args.head_C
        },
        prefer_smaller_b=True,
    )

    kron_params = sum(p.numel() for p in model.parameters())
    print(f"Kron params: {kron_params:,}  reduction {(orig_params-kron_params)/orig_params*100:.1f}%")

    # dummy forward
    print("Testing forward…")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1,20), device=device)
    out = model(input_ids=input_ids)
    print("→ logits:", out.logits.shape)
    # Test text generation
    print("\nGenerating text...")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
