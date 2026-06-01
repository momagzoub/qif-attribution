from qif_attribution.generation.backends import _disable_safety_checker


class DummyPipeline:
    def __init__(self) -> None:
        self.safety_checker = object()
        self.requires_safety_checker = True


class PipelineWithoutSafetyChecker:
    """Some pipelines (e.g. SDXL) expose no safety_checker attribute at all."""


def test_disable_safety_checker_clears_placeholder_hook() -> None:
    pipe = DummyPipeline()

    returned = _disable_safety_checker(pipe)

    assert returned is pipe
    assert pipe.safety_checker is None
    assert pipe.requires_safety_checker is False


def test_disable_safety_checker_is_noop_without_attributes() -> None:
    pipe = PipelineWithoutSafetyChecker()

    returned = _disable_safety_checker(pipe)

    assert returned is pipe
    assert not hasattr(pipe, "safety_checker")
    assert not hasattr(pipe, "requires_safety_checker")
