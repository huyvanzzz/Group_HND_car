from __future__ import annotations

from collections import defaultdict

from transformers import AutoModelForSeq2SeqLM


MODEL_ID = "google/t5-efficient-mini"


def main() -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    unique_params = sum(param.numel() for param in model.parameters())
    named_params_default = sum(param.numel() for _, param in model.named_parameters())
    named_params_with_duplicates = sum(
        param.numel() for _, param in model.named_parameters(remove_duplicate=False)
    )
    state_dict_params = sum(tensor.numel() for tensor in model.state_dict().values())

    print(f"model_id: {MODEL_ID}")
    print(f"unique_params/model.parameters(): {unique_params:,}")
    print(f"named_parameters(default): {named_params_default:,}")
    print(f"named_parameters(remove_duplicate=False): {named_params_with_duplicates:,}")
    print(f"state_dict tensor numel sum: {state_dict_params:,}")

    tied_checks = {
        "shared == lm_head": model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr(),
        "shared == encoder.embed_tokens": (
            model.shared.weight.data_ptr() == model.encoder.embed_tokens.weight.data_ptr()
        ),
        "shared == decoder.embed_tokens": (
            model.shared.weight.data_ptr() == model.decoder.embed_tokens.weight.data_ptr()
        ),
    }
    print("\ntied weight checks:")
    for name, value in tied_checks.items():
        print(f"  {name}: {value}")

    ptr_to_names: dict[int, list[str]] = defaultdict(list)
    for name, param in model.named_parameters(remove_duplicate=False):
        ptr_to_names[param.data_ptr()].append(name)

    duplicated = {ptr: names for ptr, names in ptr_to_names.items() if len(names) > 1}
    print("\nduplicated parameter names:")
    for names in duplicated.values():
        count = dict(model.named_parameters(remove_duplicate=False))[names[0]].numel()
        print(f"  {count:,}: {', '.join(names)}")


if __name__ == "__main__":
    main()
