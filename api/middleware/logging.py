"""
Logging Middleware

Logs all incoming requests and outgoing responses.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import json
import ipaddress
from typing import Callable

from database.connection import DatabaseConnection
from database.repository import AuditLogRepository

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests and responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Log request details and response status.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from next handler
        """
        # Generate request ID
        request_id = f"{time.time()}-{id(request)}"
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            f"REQUEST [{request_id}] {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Add request ID to state
        request.state.request_id = request_id
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000  # ms
            
            # Add processing time header
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            logger.info(
                f"RESPONSE [{request_id}] {response.status_code} "
                f"in {process_time:.2f}ms"
            )

            self._persist_audit_log(request, response.status_code, process_time, None)
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000  # ms
            
            # Log error
            logger.error(
                f"ERROR [{request_id}] {str(e)} "
                f"after {process_time:.2f}ms",
                exc_info=True
            )

            self._persist_audit_log(request, 500, process_time, str(e))
            
            raise

    def _persist_audit_log(
        self,
        request: Request,
        status_code: int,
        latency_ms: float,
        error_message: str | None
    ) -> None:
        """Persist request metrics to the audit_logs table for BI/operations reporting."""
        payload = None
        if request.method in {"POST", "PUT", "PATCH"}:
            try:
                body = getattr(request, "_body", None)
                if body:
                    payload = json.loads(body.decode("utf-8"))
            except Exception:
                payload = None

        try:
            client_ip = None
            if request.client and request.client.host:
                try:
                    client_ip = str(ipaddress.ip_address(request.client.host))
                except ValueError:
                    client_ip = None

            db = DatabaseConnection()
            with db.get_session() as session:
                AuditLogRepository.create(
                    session=session,
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    ip_address=client_ip,
                    error_message=error_message,
                    request_payload=payload
                )
        except Exception as exc:
            logger.warning(f"Failed to persist audit log: {exc}")
