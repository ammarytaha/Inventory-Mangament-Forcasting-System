"""
Dashboard Launcher for Fresh Flow Markets
==========================================

Simple script to launch the Streamlit dashboard.

Usage:
    python run_dashboard.py

This will start the dashboard on http://localhost:8501
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    # Get the dashboard app path
    dashboard_dir = Path(__file__).parent / "dashboard"
    app_path = dashboard_dir / "app.py"
    
    if not app_path.exists():
        print(f"Error: Dashboard app not found at {app_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("🍃 FreshFlow AI - Inventory Decision Engine")
    print("=" * 60)
    print()
    print("Starting dashboard...")
    print("Open your browser to: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")


if __name__ == "__main__":
    main()
