from pydantic import BaseModel
from .img_data import ImageData

class SearchResult(BaseModel):
    img_data: ImageData
    score: float