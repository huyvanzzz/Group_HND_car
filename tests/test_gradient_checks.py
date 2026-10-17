import torch

from efficient_vlm_ad.verification import trainable_gradient_finite_report


def test_gradient_report_counts_finite_and_non_finite_values():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    output = model(torch.ones(1, 2)).sum()
    output.backward()
    model[1].weight.grad[0, 0] = float("nan")
    model[1].bias.grad[0] = float("inf")

    report = trainable_gradient_finite_report(model)

    assert report["checked_gradient_tensors"] == 4
    assert report["parameters_without_gradient"] == []
    assert report["non_finite_gradient_tensors"] == ["1.weight", "1.bias"]
    assert report["nan_values"] == 1
    assert report["inf_values"] == 1
    assert report["global_grad_norm"] is None
    assert report["ok"] is False


def test_gradient_report_has_finite_global_norm_after_backward():
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()

    report = trainable_gradient_finite_report(model)

    assert report["ok"] is True
    assert report["checked_gradient_tensors"] == 2
    assert report["global_grad_norm"] is not None
    assert report["global_grad_norm"] > 0
