import numpy as np
from typing import Optional
from qdrant_client.http import models
from qdrant_client import QdrantClient
from config.config import configure as cf
from .img_data import ImageData
from .search_result import SearchResult
from .query_params import FilterParams
class VectorDbHandler:
    IMG_VECTOR = "image_vector"
    AVAILABLE_POINT_TYPES = [models.Record,models.ScoredPoint,models.PointStruct]

    def __init__(self) -> None:
        # self._client = qdrant_client.AsyncQdrantClient(host=cf.qdrant.host, port=cf.qdrant.port,
        #                                 grpc_port=cf.qdrant.grpc_port, api_key=cf.qdrant.api_key,
        #                                 prefer_grpc=cf.qdrant.prefer_grpc )

        # self._client = QdrantClient(host=cf.qdrant.host, port=cf.qdrant.port,
        #                                 grpc_port=cf.qdrant.grpc_port, api_key=cf.qdrant.api_key,
        #                                 url=cf.qdrant.url,
        #                                 prefer_grpc=cf.qdrant.prefer_grpc )
        
        self._client = QdrantClient(api_key=cf.qdrant.api_key,
                                        url=cf.qdrant.url)
        self.collection_name = cf.qdrant.coll
    def retrieve_by_id(self, img_id: int, with_payload = False, with_vector = False):
        res = self._client.retrieve(collection_name=self.collection_name,
                                        ids = [img_id],
                                        with_payload=with_payload,
                                        with_vectors= with_vector)
        print(res[0])
        print(len(res))
        if len(res) != 1:
            print("exist")
            return self._get_img_data_from_point(res[0])
    

    def query_search(self, query_vector,top_k = 50, filter_param: Optional[FilterParams] = None) -> "list[SearchResult]":
        res = self._client.search(collection_name=self.collection_name,
                                query_vector=(self.IMG_VECTOR,query_vector),
                                limit=top_k,
                                with_payload=True,
                                query_filter = self._get_filters_by_filter_param(filter_param))
        return [self._get_search_result_from_result_point(point) for point in res]

    def delItems(self, points):
        # points = list(range(35686))
        res =  self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points= points
            )
        )
        print(res.status)



    def get_counts(self,exact):
        res =  self._client.count(collection_name=self.collection_name, exact=exact) 
        return res.count
    def insertItems(self, item: ImageData):
        points = [self._get_point_from_img_data(item)]

        res =  self._client.upsert(collection_name=self.collection_name,
                                        wait=True,
                                        points=points)
        # print(res.status)
    

    @classmethod
    def _get_vector_from_img_data(cls,img_data: ImageData) -> models.PointVectors:
        vector = {}
        if img_data.image_vector is not None:
            vector["image_vector"] = img_data.image_vector.tolist()
        return models.PointVectors(
            id=str(img_data.id),
            vector=vector
        )
    
    @classmethod
    def _get_point_from_img_data(cls, img_data: ImageData) -> models.PointStruct:
        return models.PointStruct(
            id = img_data.id,
            payload=img_data.payload,
            vector = cls._get_vector_from_img_data(img_data).vector
        )
    @classmethod
    def _get_img_data_from_point(cls, point: models.ScoredPoint) -> ImageData:
        if point.vector:
            image_vector = np.array(point.vector[cls.IMG_VECTOR], dtype=np.float32)
        else:
            image_vector = None
        return (ImageData.from_payload(point.id,
                                        point.payload,
                                        image_vector))

    @classmethod
    def _get_search_result_from_result_point(cls,point:models.ScoredPoint) -> SearchResult:
        return SearchResult(img_data= cls._get_img_data_from_point(point) ,score=point.score)

    @staticmethod
    def _get_filters_by_filter_param(filter_param: FilterParams) -> models.Filter:
        if filter_param is None:
            return None
        
        filters = []

        if filter_param.thumbnail_list is not None:
            filters.append(models.FieldCondition(
                key="thumbnail",
                match=models.MatchAny(any=filter_param.thumbnail_list)
            ))
        if filter_param.dataset is not None:
            filters.append(models.FieldCondition(
                key="dataset",
                match = models.MatchValue(
                    value=filter_param.dataset
                )
            ))

        if filter_param.min_height is not None:
            filters.append(models.FieldCondition(
                key="height",
                range=models.Range(
                    gte=filter_param.min_height
                )
            ))

        if filter_param.min_width is not None:
            filters.append(models.FieldCondition(
                key="width",
                range=models.Range(
                    gte=filter_param.min_width
                )
            ))

        if filter_param.min_aspect_ratio is not None:
            filters.append(models.FieldCondition(
                key="aspect_ratio",
                range=models.Range(
                    gte=filter_param.min_aspect_ratio,
                    lte=filter_param.max_aspect_ratio
                )
            ))

        if not filters:
            return None
        return models.Filter(
            must= filters
        )