class DetectInfo():
    def __init__(self, label: str, box: tuple[int], score: float):
        self.label = label
        self.box = box
        self.score = score
        self.size = -1
        self.right_arrow = False
        self.left_arrow = False

    def set_size(self, size: int):
        self.size = size
