class BoundingBox:

    def __init__(self, frame, id, label, xtl, ytl, xbr, ybr, confidence = 0):
        self.frame = frame
        self.id = id
        self.label = label
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.confidence = confidence

    @property
    def bbox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    @property
    def width(self):
        return abs(self.xbr - self.xtl)

    @property
    def height(self):
        return abs(self.ybr - self.ytl)

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (int((self.xtl + self.xbr) / 2), int((self.ytl + self.ybr) / 2))

    def __str__(self):
        return f'frame={self.frame}, id={self.id}, label={self.label}, bbox={self.bbox}, confidence={self.confidence}'