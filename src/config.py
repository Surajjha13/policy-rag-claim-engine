from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    llm_provider: str = "groq"
    llm_api_key: str = ""
    llm_model: str = "groq/llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    index_dir: str = "./index_store"
    policy_pdf_path: str = "./data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_reranked: int = 4
    validation_fail_retry_limit: int = 1


settings = Settings()
