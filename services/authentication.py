"""
OrthoTwin AI — Authentication Service
Implements secure cryptographic hashing (PBKDF2-HMAC-SHA256 with 100,000 iterations),
salt generation, credential verification, and safe user provisioning.
No universal hard-coded passwords or insecure default credentials.
"""

import hashlib
import secrets
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from database.models import User, Role
from database.repositories import UserRepository
from config import HASH_ITERATIONS, HASH_ALGORITHM


def generate_salt(length: int = 16) -> str:
    """Generates a cryptographically strong random hex salt."""
    return secrets.token_hex(length)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a plaintext password using PBKDF2-HMAC-SHA256 with 100,000 rounds.
    Returns (hex_password_hash, hex_salt).
    """
    if not password:
        raise ValueError("Password cannot be empty")
    if not salt:
        salt = generate_salt()
    pwd_bytes = password.encode("utf-8")
    salt_bytes = bytes.fromhex(salt)
    dk = hashlib.pbkdf2_hmac(
        HASH_ALGORITHM,
        pwd_bytes,
        salt_bytes,
        HASH_ITERATIONS
    )
    return dk.hex(), salt


def verify_password(plain_password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plaintext password against a stored PBKDF2 hash using constant-time comparison."""
    if not plain_password or not stored_hash or not salt:
        return False
    try:
        calculated_hash, _ = hash_password(plain_password, salt)
        return secrets.compare_digest(calculated_hash, stored_hash)
    except Exception:
        return False


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    Authenticates a user by username and password.
    Returns the User object if valid and active, otherwise None.
    """
    user = UserRepository.get_by_username(db, username.strip())
    if not user or not user.is_active:
        return None
    if verify_password(password, user.password_hash, user.salt):
        return user
    return None


def create_development_user(
    db: Session,
    username: str,
    password: str,
    role_name: str,
    full_name: str,
    email: Optional[str] = None,
    patient_id: Optional[int] = None
) -> User:
    """
    Explicit development-account creation function.
    Validates minimum password length and creates salted record.
    """
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    existing = UserRepository.get_by_username(db, username)
    if existing:
        raise ValueError(f"Username '{username}' already exists")

    pwd_hash, salt = hash_password(password)
    return UserRepository.create_user(
        db=db,
        username=username,
        password_hash=pwd_hash,
        salt=salt,
        role_name=role_name,
        full_name=full_name,
        email=email,
        patient_id=patient_id
    )
