import argparse
import asyncio
from config.config import configure as cf
from extract_features import qdrant_index
from qdrant.qdrant_create_collection import create_cloud_collection
from qdrant.provider import db_handler
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--init-qdrant',dest='init_qdrant',action='store_true',help="Initiate Qdrant db")

    parser.add_argument('--qdrant-index',dest= "qdrant_index", help="store feature in qdrant", nargs=2, type=int)
    parser.add_argument('--delete-index',dest='delete_index',action='store_true')
    parser.add_argument('--retrieve-id',dest='retrieve_id',type=int)

    return parser.parse_args()
if __name__ == '__main__':
    

    args = parse_args()
    if args.init_qdrant:
        create_cloud_collection(cf.qdrant.host, cf.qdrant.port, cf.qdrant.coll)

    elif args.qdrant_index:
        param1 = args.qdrant_index[0] if len(args.qdrant_index) > 0 else 0
        param2 = args.qdrant_index[1] if len(args.qdrant_index) > 1 else 0
        print(param1, param2)
        qdrant_index(param1, param2)

    elif args.delete_index:
        points = list(range(4611,4612))

        db_handler.delItems(points)
    
    elif args.retrieve_id is not None:
        print(args.retrieve_id)
        img_data = db_handler.retrieve_by_id(args.retrieve_id,True,True)
        print(img_data.id)
        print(img_data.thumbnail)


