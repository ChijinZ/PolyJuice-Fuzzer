from nnsmith.backends.pt2 import PT2


class Hidet(PT2):
    def __init__(self, target: str = "cpu", optmax: bool = True, **kwargs):
        if target != "cuda":
            raise ValueError("Hidet backend only supports GPU!")
        super().__init__(target, optmax, backend="hidet", **kwargs)

    @property
    def system_name(self) -> str:
        return "hidet"
