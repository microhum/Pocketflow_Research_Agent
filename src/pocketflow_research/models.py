from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime
from numpy import ndarray

FaissIndexObject = Any

class PaperMetadataBase(BaseModel):
    title: str
    source_url: Optional[str] = None
    source: str
    authors: Optional[List[str]] = Field(default_factory=list)
    published_date: Optional[Union[str, datetime]] = None
    abstract: Optional[str] = None
    id: Optional[Union[str, int]] = None

    class Config:
        arbitrary_types_allowed = True

class FetchedPaperInfo(BaseModel):
    title: str
    source_url: Optional[str] = None
    source: str # "arxiv" or "thaijo"
    authors: Optional[List[str]] = Field(default_factory=list)
    published_date: Optional[Union[str, datetime]] = None

    class Config:
        arbitrary_types_allowed = True

class ChunkSourceInfo(BaseModel):
    title: str
    source: str

class RetrievedChunk(BaseModel):
    text: str
    original_source: ChunkSourceInfo

class SharedStore(BaseModel):
    topic: Optional[str] = None
    query_intent: Optional[Literal["FETCH_NEW", "QA_CURRENT", "unknown"]] = None
    search_keywords: Optional[str] = None
    fetched_papers: List[FetchedPaperInfo] = Field(default_factory=list)
    temp_embeddings: Optional[Any] = None 
    chunk_source_map: Dict[int, ChunkSourceInfo] = Field(default_factory=dict)
    temp_embeddings: Optional[ndarray] = None
    query_embedding: Optional[Any] = None
    temp_index: Optional[FaissIndexObject] = None
    query_embedding: Optional[ndarray] = None
    retrieved_chunks_with_source: List[RetrievedChunk] = Field(default_factory=list)
    answer_text: Optional[str] = None
    answer_sources: List[ChunkSourceInfo] = Field(default_factory=list)
    final_answer: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True # To allow np.ndarray and FaissIndexObject

class ArxivAuthorSchema(BaseModel):
    name: str

class ArxivPaperSchema(BaseModel):
    entry_id: str
    title: str
    authors: List[ArxivAuthorSchema]
    summary: str
    published: datetime
    updated: datetime
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    primary_category: Optional[str] = None
    categories: Optional[List[str]] = Field(default_factory=list)

    @classmethod
    def from_arxiv_result(cls, result: Any):
        return cls(
            entry_id=result.entry_id,
            title=result.title,
            authors=[ArxivAuthorSchema(name=author.name) for author in result.authors],
            summary=result.summary,
            published=result.published,
            updated=result.updated,
            pdf_url=result.pdf_url,
            doi=result.doi,
            primary_category=result.primary_category,
            categories=result.categories
        )

class ThaiJoAuthorSchema(BaseModel):
    author_id: Optional[int] = None
    full_name: Dict[str, Optional[str]]
    affiliation: Optional[Dict[str, Optional[str]]] = None

class ThaiJoPaperSchema(BaseModel):
    id: int
    title: str
    abstract: Optional[str] = None
    authors: List[str]
    url: Optional[str] = None
    published_date: Optional[str] = None
    source: Literal["thaijo"]

    class Config:
        arbitrary_types_allowed = True
