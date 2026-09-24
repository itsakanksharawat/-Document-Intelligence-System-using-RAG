from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from src.config import GOOGLE_API_KEY


PASSWORD_HASHER = PasswordHash.recommended()

JWT_ALGORITHM = "HS256"

# Temporary default; we will read the real value from .env below.
JWT_SECRET_KEY = None


def hash_password(password: str) -> str:
    return PASSWORD_HASHER.hash(password)


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    return PASSWORD_HASHER.verify(
        password,
        password_hash,
    )


def create_access_token(
    user_id: int,
    expires_minutes: int = 60,
) -> str:
    if JWT_SECRET_KEY is None:
        raise ValueError(
            "JWT_SECRET_KEY is not configured."
        )

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes
    )

    payload = {
        "sub": str(user_id),
        "exp": expire,
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )