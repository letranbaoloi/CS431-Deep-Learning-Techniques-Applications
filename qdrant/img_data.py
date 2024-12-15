from pydantic import BaseModel, Field, computed_field
from typing import Optional, List
from numpy import ndarray
from datetime import datetime

class ImageData(BaseModel):
    id: int
    thumbnail: Optional[str] = None
    image_vector: Optional[ndarray] = Field(None, exclude=True)
    index_date: datetime
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[float] = None
    dataset: Optional[str] = None

    @property
    def payload(self):
        # tao ra dict cua model
        result = self.model_dump(exclude={'id','index_date'})
        result['index_date'] = self.index_date.isoformat()
        return result
    
    @classmethod
    def from_payload(cls, id:int, payload: dict, image_vector:Optional[ndarray]=None):
        index_date = datetime.fromisoformat(payload['index_date'])
        del payload['index_date']

        return cls(id = id,
                    index_date = index_date,
                    **payload,
                    image_vector = image_vector if image_vector is not None else None)
    
    class Config:
        arbitrary_types_allowed = True