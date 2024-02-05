def print_args(args: Tuple[Any]) -> None:
    print(f"Positional args (count: {len(args)})")
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            arg = f"Tensor{tuple(arg.shape)}"
        print(f"  - arg({i}): {arg}")

def print_kwargs(kwargs: Dict[str, Any]) -> None:
    print(f"Keyword args (count: {len(kwargs)})")
    for i, (kwarg_name, kwarg_value) in enumerate(kwargs.items()):
        if isinstance(kwarg_value, torch.Tensor):
            kwarg_value = f"Tensor{tuple(kwarg_value.shape)}"
        print(f"  - kwarg({i}): {kwarg_name}={kwarg_value}")

def print_output(output: Any) -> None:
    if isinstance(output, torch.Tensor):
        print(f"Output: Tensor{tuple(output.shape)}")
    elif isinstance(output, BaseModelOutputWithPooling):
        print(f"Output type: {type(output).__name__}")
        print(f"  last_hidden_state=Tensor{tuple(output.last_hidden_state.shape)}")
        print(f"  pooler_output=Tensor{tuple(output.pooler_output.shape)}")
    elif isinstance(output, UNet2DConditionOutput):
        print(f"Output type: {type(output).__name__}")
        print(f"  sample=Tensor{tuple(output.sample.shape)}")
    else:
        print(f"Output type: {type(output).__name__}")

def make_forward_verbose(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """
    The `make_forward_verbose` function is implemented as a Python decorator function with custom arguments.
    The `make_forward_verbose` decorates an input model.
    Model decoration consists in:
        1. Decorating the model's forward method using the `make_verbose` decorator function,
        2. Monkey-patching the orginal method with the decorated one.
    """
    def make_verbose(f: Callable) -> Callable:
        def decorated_forward_method(*args, **kwargs) -> Any:
            print("-"*50)
            print(f"Model: {model_name}")
            print(f"Model type: {type(model).__name__}")
            print_args(args)
            print_kwargs(kwargs)
            output = f(*args, **kwargs)
            print_output(output)
            return output
        return decorated_forward_method
    model.forward = make_verbose(model.forward)
    return model