from PIL import Image


class RatioScale:
    def __init__(self, max_size, method=Image.ANTIALIAS):
        self.max_size = max_size
        self.method = method

    def __call__(self, image):
        image.thumbnail(self.max_size, self.method)
        offset = (int((self.max_size[0] - image.size[0]) / 2), int((self.max_size[1] - image.size[1]) / 2))
        back = Image.new("RGB", self.max_size, "black")
        back.paste(image, offset)
        return back
