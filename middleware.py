from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from fastapi import Request
import logging

logger = logging.getLogger(__name__)

class HTTPSProxyMiddleware(BaseHTTPMiddleware):
    """Middleware to handle HTTPS for proxied applications like Hugging Face Spaces."""
    
    async def dispatch(self, request: Request, call_next):
        """
        Dispatches the middleware.
        
        1. Modifies the request headers to ensure proper URL scheme
        2. Logs details about the request for debugging
        """
        # Log original request info for debugging
        logger.info(f"Original request: scheme={request.url.scheme}, url={request.url}")
        
        # Detect if we're behind a proxy that's forwarding HTTPS as HTTP
        forwarded_proto = request.headers.get("X-Forwarded-Proto")
        
        # If we have a forwarded proto header and it's HTTPS, 
        # modify the request's scope to ensure URLs are generated with HTTPS
        if forwarded_proto == "https" and request.url.scheme == "http":
            logger.info("Detected HTTPS behind proxy, updating request scope")
            request.scope["scheme"] = "https"
        
        # Continue processing the request
        response = await call_next(request)
        return response 