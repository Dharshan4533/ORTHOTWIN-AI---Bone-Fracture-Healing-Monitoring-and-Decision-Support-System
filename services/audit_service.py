"""
OrthoTwin AI — Audit Logging Service
Tracks clinical actions, security events, and data modifications
without logging sensitive credentials or unnecessary private health details.
"""

from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from database.models import User, AuditLog
from database.repositories import AuditRepository


def log_audit_event(
    db: Session,
    action: str,
    user: Optional[User] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None
) -> AuditLog:
    """Records an immutable audit trail entry in the database."""
    user_id = user.id if user else None
    return AuditRepository.log(
        db=db,
        action=action,
        user_id=user_id,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details,
        ip_address=ip_address
    )
