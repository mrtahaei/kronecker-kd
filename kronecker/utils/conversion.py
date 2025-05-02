#!/usr/bin/env python
import math
import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from kronecker.layers.kronecker_linear import KroneckerLinear


def _get_parent_and_attr(model: nn.Module, full_name: str):
    """Given 'encoder.layer.3.attention.self.query', return (parent_module, 'query')."""
    *path, attr = full_name.split('.')
    parent = model
    for p in path:
        # normal attribute
        if hasattr(parent, p):
            parent = getattr(parent, p)
        # indexing into ModuleList / list / tuple
        elif p.isdigit() and isinstance(parent, (list, tuple, nn.ModuleList)):
            parent = parent[int(p)]
        else:
            raise AttributeError(f"Cannot traverse '{p}' in '{full_name}'")
    return parent, attr


def divisors(x: int):
    """Yield all positive divisors of x."""
    for i in range(1, int(math.sqrt(x)) + 1):
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
) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """
    Find (m1,m2),(n1,n2) such that
      m1*m2 == out_features,
      n1*n2 == in_features,
    and
      (in_features*out_features) / (m1*n1 + m2*n2) >= compression.

    If prefer_smaller_b=True, ties are broken by minimizing m2*n2 (the size of B).
    """
    orig = in_features * out_features
    candidates = []

    # iterate all factorizations
    for m2 in divisors(out_features):
        m1 = out_features // m2
        # enforce B no larger than A
        if m2 > m1:
            continue
        for n2 in divisors(in_features):
            n1 = in_features // n2
            if n2 > n1:
                continue

            kron_params = m1 * n1 + m2 * n2
            ratio = orig / kron_params
            if ratio >= compression:
                b_size = m2 * n2
                candidates.append((kron_params, b_size, (m1, m2), (n1, n2)))

    if not candidates:
        raise ValueError(
            f"No valid Kronecker factors for compression={compression:.2f} "
            f"on ({out_features}×{in_features})"
        )

    # pick best candidate
    if prefer_smaller_b:
        # first by B size, then by total params
        _, _, out_fac, in_fac = min(candidates, key=lambda x: (x[1], x[0]))
    else:
        # just minimize total kron-params
        _, _, out_fac, in_fac = min(candidates, key=lambda x: x[0])

    return out_fac, in_fac



def replace_linears_with_kron(
    model: PreTrainedModel,
    compression: dict = None,
):
    compression = compression or {'attention': 4.0, 'ffn': 8.0, 'head': 2.0}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        breakpoint()
        lname = name.lower()
        if 'attention' in lname or 'attn' in lname:
            C = compression['attention']
        elif 'ffn' in lname or 'mlp' in lname or 'intermediate' in lname:
            C = compression['ffn']
        elif 'lm_head' in lname :
            C = compression['head']
        else:
            continue

        (m1, m2), (n1, n2) = compute_kron_factors(
            module.in_features, module.out_features, C
        )
        print(f"[{name}] → out_factors=({m1},{m2}), in_factors=({n1},{n2})")

        kron_layer = KroneckerLinear.from_linear_with_factors(
            module,
            out_factors=(m1, m2),
            in_factors=(n1, n2),
            num_sum=1,
            efficient_sum=True,
        )

        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, kron_layer)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Test Kronecker conversion")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--attention_compression", type=float, default=4.0)
    parser.add_argument("--ffn_compression", type=float, default=8.0)
    parser.add_argument("--head_compression", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_name} → {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {orig_params:,}")

    replace_linears_with_kron(
        model,
        compression={
            "attention": args.attention_compression,
            "ffn":       args.ffn_compression,
            "head":      args.head_compression,
        },
    )

    kron_params = sum(p.numel() for p in model.parameters())
    reduction = (orig_params - kron_params) / orig_params * 100
    print(f"Kronecker parameters: {kron_params:,} ({reduction:.1f}% reduction)")

    # dummy forward
    print("Running dummy forward…")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 20), device=device)
    try:
        out = model(input_ids=input_ids)
        print("Success! logits shape:", out.logits.shape)
    except Exception as e:
        print("Forward failed:", e)


if __name__ == "__main__":
    main()
