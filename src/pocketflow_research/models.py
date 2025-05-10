from typing import List, Optional, Dict, Any, Literal, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

FaissIndexObject = Any

# New model for chat messages
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

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

# Define the every node types as a literal type
NodeTypes = Literal["GetTopicNode", "QueryIntentClassifierNode", "KeywordExtractorNode", 
                    "FetchArxivNode", "FetchThaijoNode", "ProcessPapersBatchNode", 
                    "EmbedChunksNode", "BuildTempIndexNode", "EmbedQueryNode", 
                    "RetrieveChunksNode", "GenerateResponseNode", 
                    "CiteSourcesNode", "EarlyEndReporterNode"]
class SharedStore(BaseModel):
    topic: Optional[str] = None # Explicitly for embedding and search
    chat_history: List[ChatMessage] = Field(default_factory=list) # For conversational context
    system_instructions: Optional[str] = None # For system messages
    query_intent: Optional[Literal["fetch_new", "qa_current", "unknown"]] = None
    current_node: Optional[NodeTypes] = None # For tracking current node in flow
    early_end_reason: Optional[str] = None
    search_keywords: Optional[str] = None # May be derived from topic or chat_history
    fetched_papers: List[FetchedPaperInfo] = Field(default_factory=list)
    temp_embeddings: Optional[Any] = None
    chunk_source_map: Dict[int, ChunkSourceInfo] = Field(default_factory=dict)
    query_embedding: Optional[Any] = None
    temp_index: Optional[FaissIndexObject] = None
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

class ThaiJoPaperSchema(TypedDict):
    id: int
    title: str # Note: In ThaiJO API response, title is an object {"en_US": "...", "th_TH": "..."}
              # This model simplifies it to a single string. Consider adjusting if specific language needed.
    abstract: str # Similar to title, abstract is an object in API response.
    authors: List[str] # Simplified. API provides more structured author info.
    url: str
    published_date: str
    source: Literal["thaijo"]
