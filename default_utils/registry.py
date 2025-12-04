
METRICS_FUNCTIONS: dict[str, callable] = dict()
def register_metric(name: str):
    def decorator(func):
        METRICS_FUNCTIONS[name] = func
        return func
    return decorator


CONFIDENCE_FUNCTIONS: dict[str, callable] = dict()
def register_confidence(name: str):
    def decorator(func):
        CONFIDENCE_FUNCTIONS[name] = func
        return func
    return decorator


GRADER_FUNCTIONS: dict[str, callable] = dict()
def register_grader(name: str):
    def decorator(func):
        GRADER_FUNCTIONS[name] = func
        return func
    return decorator


PROMPT_FORMATTER: dict[str, callable] = dict()
def register_prompt_formatter(name: str):
    def decorator(func):
        PROMPT_FORMATTER[name] = func
        return func
    return decorator


FILTER_FUNCTIONS: dict[str, callable] = dict()
def register_filter(name: str):
    def decorator(func):
        FILTER_FUNCTIONS[name] = func
        return func
    return decorator