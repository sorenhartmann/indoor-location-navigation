# Contains trace data
class TraceData:
    pass


# Contains the corresponding tensor?
class TraceTensor:
    pass


# Loads all?
class LocationDataset:
    def __init__(self, site=None, floor=None, test=False) -> None:

        if site is None and floor is None and test is False:
            raise ValueError

        pass