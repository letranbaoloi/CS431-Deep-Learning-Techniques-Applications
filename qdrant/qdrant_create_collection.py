
from qdrant_client import qdrant_client, models
from .provider import db_handler

# def creat_collection(host,port,name):
#     client = qdrant_client.QdrantClient(host = host, port= port)
#     vectors_config = {"image_vector": models.VectorParams(size=640, distance=models.Distance.COSINE),
    
#     }
#     client.create_collection(collection_name=name,
#                             vectors_config=vectors_config)
#     print("coll create")


def create_cloud_collection(host,port,name,test = False):
    remote_client = qdrant_client.QdrantClient(
        url="https://17a9a619-1117-40ce-a7a3-de9d44485164.us-west-1-0.aws.cloud.qdrant.io:6333", 
        api_key="HvcaV1q_ABt2tgqISulnQNvLxAUWk6T7YaAwZiHGL8rBaomw_uXx3A",
    )
    vectors_config = {"image_vector": models.VectorParams(size=640, distance=models.Distance.COSINE)
    }
    remote_client.recreate_collection(collection_name=name,vectors_config=vectors_config)

    if test == True:
        local_client = qdrant_client.QdrantClient(host = host, port= port)

        res = db_handler.retrieve_by_id(1,True,True)
        remote_client.upload_records(name,res)

        