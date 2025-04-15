"""
Main entry point for the hurricane prediction API
"""

import os
import uvicorn
import logging
from hurricane_agents.api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hurricane_agents.main')

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Log startup message
    logger.info(f"Starting Hurricane Prediction API on port {port}")
    
    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "hurricane_agents.api:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )