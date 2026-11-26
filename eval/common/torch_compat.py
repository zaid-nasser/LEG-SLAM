import torch
import inspect

def patch_torch_load_state_dict_compat() -> None:
    """
    Compatibility for PyTorch 2.0.x.

    Transformers 4.50 passes `assign=` to nn.Module.load_state_dict(),
    but PyTorch 2.0.x does not support that argument.

    For this RADIO/SigLIP loading path the model is normally initialized,
    not a meta-device model, so we safely fall back to the traditional
    copy-based load_state_dict behavior.
    """
    import inspect

    original = torch.nn.Module.load_state_dict

    if "assign" in inspect.signature(original).parameters:
        return

    def compatible_load_state_dict(
        self,
        state_dict,
        strict=True,
        **kwargs,
    ):
        # PyTorch 2.0 has no `assign` option.
        # Drop it and use the traditional copy-based behavior.
        kwargs.pop("assign", None)

        if kwargs:
            raise TypeError(
                f"Unexpected load_state_dict arguments: "
                f"{list(kwargs.keys())}"
            )

        return original(
            self,
            state_dict,
            strict=strict,
        )

    torch.nn.Module.load_state_dict = compatible_load_state_dict

    print(
        "Applied PyTorch compatibility patch: "
        "nn.Module.load_state_dict() does not support assign; "
        "using copy-based loading."
    )


def patch_torch_compiler_compat() -> None:
    """
    Compatibility for PyTorch 2.0.x.

    Transformers 4.50 imports flex_attention.py, which references
    torch.compiler.disable even when Flex Attention is not used.

    RADIO's SigLIP2 text path does not use Flex Attention, so a no-op
    decorator is sufficient for this evaluation path.
    """
    if hasattr(torch, "compiler"):
        return

    class _CompilerCompat:
        @staticmethod
        def disable(fn=None, recursive=True):
            if fn is None:
                def decorator(func):
                    return func
                return decorator
            return fn

        @staticmethod
        def is_compiling():
            return False

    torch.compiler = _CompilerCompat()

    print(
        "Applied PyTorch compatibility patch: "
        "torch.compiler namespace is unavailable."
    )


def patch_torch_hub_weights_only_compat() -> None:
    """
    Compatibility for older PyTorch versions.

    Current RADIO passes `weights_only=False` to
    torch.hub.load_state_dict_from_url(), while older PyTorch versions
    do not expose that keyword argument.
    """
    original = torch.hub.load_state_dict_from_url

    if "weights_only" in inspect.signature(original).parameters:
        return

    def compatible_load_state_dict_from_url(*args, **kwargs):
        kwargs.pop("weights_only", None)
        return original(*args, **kwargs)

    torch.hub.load_state_dict_from_url = compatible_load_state_dict_from_url

    print(
        "Applied PyTorch compatibility patch: "
        "load_state_dict_from_url() does not support weights_only."
    )

