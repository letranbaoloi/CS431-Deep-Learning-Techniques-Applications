from pydantic import BaseModel
# from pydantic_settings import BaseSettings, SettingsConfigDict

class QdrantSettings(BaseModel):
    host: str = 'localhost'
    port: int = 6333
    grpc_port: int = 6334 # src: qdrant docs
    coll: str = 'clip_coll'
    prefer_grpc: bool = False
    api_key: str = 'HvcaV1q_ABt2tgqISulnQNvLxAUWk6T7YaAwZiHGL8rBaomw_uXx3A'
    url: str = 'https://17a9a619-1117-40ce-a7a3-de9d44485164.us-west-1-0.aws.cloud.qdrant.io:6333'
class ClipSetting(BaseModel):
    model: str = 'RN50x4'

class Config():
    qdrant: QdrantSettings = QdrantSettings()
    clip: ClipSetting = ClipSetting()


configure = Config()