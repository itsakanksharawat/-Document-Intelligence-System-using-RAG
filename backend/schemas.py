from pydantic import BaseModel


class CompanyCreate(BaseModel):
    name: str
    company_key: str


class CompanyResponse(BaseModel):
    id: int
    name: str
    company_key: str

    class Config:
        from_attributes = True
class DocumentResponse(BaseModel):
    id: int
    company_id: int
    document_name: str
    document_type: str
    file_type: str
    source_type: str | None = None
    file_path: str
    content_hash: str | None = None

    class Config:
        from_attributes = True
class RegisterRequest(BaseModel):
    company_name: str
    company_key: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str