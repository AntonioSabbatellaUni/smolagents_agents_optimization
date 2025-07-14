"""Session management for organizing experiment outputs."""

import os
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class SessionManager:
    """Manages experiment sessions and output folders."""
    
    def __init__(self, base_experiments_dir: str = "experiments"):
        """Initialize session manager.
        
        Args:
            base_experiments_dir: Base directory for experiments
        """
        self.base_dir = Path(base_experiments_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_session_dir: Optional[Path] = None
        self.session_info: Dict[str, Any] = {}
    
    def create_session(self, 
                      experiment_id: str = "default",
                      question: str = "",
                      custom_suffix: str = "") -> Path:
        """Create a new session directory.
        
        Args:
            experiment_id: Base experiment identifier
            question: The question being asked (for metadata)
            custom_suffix: Custom suffix for the directory name
            
        Returns:
            Path to the created session directory
        """
        # Create timestamp-based folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if custom_suffix:
            folder_name = f"{experiment_id}_{timestamp}_{custom_suffix}"
        else:
            folder_name = f"{experiment_id}_{timestamp}"
        
        session_dir = self.base_dir / folder_name
        
        # Handle existing directory by adding increment
        counter = 1
        original_session_dir = session_dir
        while session_dir.exists():
            session_dir = Path(f"{original_session_dir}_{counter}")
            counter += 1
        
        session_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_dir = session_dir
        
        # Store session metadata
        self.session_info = {
            "experiment_id": experiment_id,
            "session_dir": str(session_dir),
            "timestamp": timestamp,
            "question": question,
            "created_at": datetime.now().isoformat(),
            "folder_name": folder_name
        }
        
        return session_dir
    
    def get_session_dir(self) -> Optional[Path]:
        """Get the current session directory.
        
        Returns:
            Path to current session directory or None if no session
        """
        return self.current_session_dir
    
    def save_config_copy(self, config_file_path: str):
        """Save a copy of the configuration file to session directory.
        
        Args:
            config_file_path: Path to the configuration file to copy
        """
        if not self.current_session_dir:
            raise ValueError("No active session. Call create_session first.")
        
        config_path = Path(config_file_path)
        if config_path.exists():
            shutil.copy(config_path, self.current_session_dir / config_path.name)
    
    def get_file_path(self, filename: str) -> Path:
        """Get full path for a file in the current session directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Full path to the file in the session directory
        """
        if not self.current_session_dir:
            raise ValueError("No active session. Call create_session first.")
        
        return self.current_session_dir / filename
    
    def list_sessions(self, experiment_id: Optional[str] = None) -> list:
        """List all session directories.
        
        Args:
            experiment_id: Filter by experiment ID, or None for all
            
        Returns:
            List of session directory paths
        """
        sessions = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                if experiment_id is None or item.name.startswith(experiment_id):
                    sessions.append(item)
        
        return sorted(sessions, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session.
        
        Returns:
            Dictionary with session information
        """
        return self.session_info.copy()
    
    def cleanup_old_sessions(self, keep_count: int = 10, experiment_id: Optional[str] = None):
        """Clean up old session directories.
        
        Args:
            keep_count: Number of recent sessions to keep
            experiment_id: Only cleanup sessions for this experiment ID, or None for all
        """
        sessions = self.list_sessions(experiment_id)
        
        if len(sessions) > keep_count:
            sessions_to_remove = sessions[keep_count:]
            for session_dir in sessions_to_remove:
                shutil.rmtree(session_dir)
                print(f"Removed old session: {session_dir}")
    
    @staticmethod
    def get_session_dir_from_path(experiments_dir: str = "experiments") -> Optional[Path]:
        """Get the most recent session directory.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            Path to the most recent session directory or None
        """
        base_dir = Path(experiments_dir)
        if not base_dir.exists():
            return None
        
        sessions = []
        for item in base_dir.iterdir():
            if item.is_dir():
                sessions.append(item)
        
        if not sessions:
            return None
        
        return sorted(sessions, key=lambda x: x.stat().st_mtime, reverse=True)[0]
