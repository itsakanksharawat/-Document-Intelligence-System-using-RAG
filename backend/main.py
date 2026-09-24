from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from backend.rag_service import load_rag_system
from src.rag import ask_rag
from sqlalchemy import text
from backend.database import get_db
from fastapi import Depends

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Company
from backend.schemas import CompanyCreate, CompanyResponse

from backend.models import Company, Document,User
from backend.schemas import (
    CompanyCreate,
    CompanyResponse,
    DocumentResponse,
)
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Company, Document

from src.config import DATA_DIR
from src.ingestion import ingest_documents
from src.vector_store import create_index

from backend.auth import (
    hash_password,
    verify_password,
    create_access_token,
)

from backend.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
)

app = FastAPI(
    title="Company Knowledge Copilot API",
    description="Backend API for the RAG-powered company document intelligence system.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    company_id: str
    question: str


rag_index = None
rag_reranker = None
rag_llm = None


@app.on_event("startup")
def startup_event():
    global rag_index
    global rag_reranker
    global rag_llm

    (
        rag_index,
        rag_reranker,
        rag_llm,
    ) = load_rag_system()


@app.get("/")
def root():
    return {
        "message": "Company Knowledge Copilot API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    if rag_index is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not loaded.",
        )

    result = ask_rag(
        question=request.question,
        index=rag_index,
        company_id=request.company_id,
        reranker=rag_reranker,
        llm=rag_llm,
    )

    return {
        "question": result["question"],
        "company_id": result["company_id"],
        "answerable": result["answerable"],
        "answer": result["answer"],
        "sources": result["sources"],
    }
@app.get("/database/health")
def database_health(db=Depends(get_db)):
    result = db.execute(text("SELECT 1"))
    value = result.scalar()

    return {
        "database": "PostgreSQL",
        "status": "connected",
        "test": value,
    }
@app.post(
    "/companies",
    response_model=CompanyResponse,
)
def create_company(
    company: CompanyCreate,
    db: Session = Depends(get_db),
):
    new_company = Company(
        name=company.name,
        company_key=company.company_key,
    )

    db.add(new_company)
    db.commit()
    db.refresh(new_company)

    return new_company


@app.get(
    "/companies",
    response_model=list[CompanyResponse],
)
def get_companies(
    db: Session = Depends(get_db),
):
    return db.query(Company).all()
@app.get(
    "/companies/{company_id}/documents",
    response_model=list[DocumentResponse],
)
def get_company_documents(
    company_id: int,
    db: Session = Depends(get_db),
):
    company = (
        db.query(Company)
        .filter(Company.id == company_id)
        .first()
    )

    if company is None:
        raise HTTPException(
            status_code=404,
            detail="Company not found.",
        )

    return (
        db.query(Document)
        .filter(Document.company_id == company_id)
        .all()
    )
@app.post("/companies/{company_id}/documents/upload")
async def upload_company_document(
    company_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Find company
    company = (
        db.query(Company)
        .filter(Company.id == company_id)
        .first()
    )

    if company is None:
        raise HTTPException(
            status_code=404,
            detail="Company not found.",
        )

    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    # Validate file type
    if extension not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    # Company-specific directory
    company_dir = (
        Path(DATA_DIR)
        / company.company_key
    )

    company_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = company_dir / filename

    # Prevent accidental replacement
    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="Document already exists.",
        )

    # Save uploaded file
    file_content = await file.read()

    with open(destination, "wb") as output_file:
        output_file.write(file_content)

    try:
        # Re-run ingestion
        nodes = ingest_documents()

        # Add new/modified documents to ChromaDB
        create_index(nodes)

        # Find metadata for uploaded document
        uploaded_metadata = None

        for node in nodes:
            metadata = node.metadata

            if (
                metadata.get("company_id")
                == company.company_key
                and metadata.get("document_name")
                == filename
            ):
                uploaded_metadata = metadata
                break

        if uploaded_metadata is None:
            raise RuntimeError(
                "Document was saved but metadata "
                "could not be created."
            )

        # Store metadata in PostgreSQL
        document = Document(
            company_id=company.id,
            document_name=filename,
            document_type=uploaded_metadata.get(
                "document_type",
                "unknown",
            ),
            file_type=extension.lstrip("."),
            source_type=uploaded_metadata.get(
                "source_type",
            ),
            file_path=str(destination),
            content_hash=uploaded_metadata.get(
                "content_hash",
            ),
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        return {
            "message": "Document uploaded and indexed successfully.",
            "document_id": document.id,
            "company_id": company.id,
            "company_key": company.company_key,
            "document_name": document.document_name,
            "document_type": document.document_type,
            "file_type": document.file_type,
            "source_type": document.source_type,
        }

    except Exception as exc:
        db.rollback()

        # Remove the physical file if processing failed
        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(exc)}",
        )
@app.post("/auth/register")
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    email = request.email.strip().lower()
    company_key = request.company_key.strip().lower()
    company_name = request.company_name.strip()

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email is required.",
        )

    if len(request.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters.",
        )

    if not company_name:
        raise HTTPException(
            status_code=400,
            detail="Company name is required.",
        )

    if not company_key:
        raise HTTPException(
            status_code=400,
            detail="Company key is required.",
        )

    existing_user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="User with this email already exists.",
        )

    existing_company = (
        db.query(Company)
        .filter(Company.company_key == company_key)
        .first()
    )

    if existing_company:
        raise HTTPException(
            status_code=409,
            detail="Company key already exists.",
        )

    company = Company(
        name=company_name,
        company_key=company_key,
    )

    db.add(company)
    db.flush()

    user = User(
        company_id=company.id,
        email=email,
        password_hash=hash_password(request.password),
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    db.refresh(company)

    return {
        "message": "Registration successful.",
        "user_id": user.id,
        "company_id": company.id,
        "company_key": company.company_key,
    }
@app.post(
    "/auth/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    email = request.email.strip().lower()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    if not verify_password(
        request.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    access_token = create_access_token(
        user_id=user.id,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }