class FilterParams:
    def __init__(self, thumbnail_list = None, dataset = None, min_width  = None,min_height = None, min_aspect_ratio = None,max_aspect_ratio = None):
        self.thumbnail_list = thumbnail_list
        self.dataset = dataset
        self.min_width = min_width
        self.min_height = min_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
