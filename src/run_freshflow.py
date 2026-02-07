"""
FreshFlow AI - Solution Runner
==============================

Simple script to run the FreshFlow AI dashboard.

Usage:
    python run_freshflow.py [--port PORT]
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'pandas', 'plotly', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
            
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\n📦 Installing dependencies...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            '-r', str(Path(__file__).parent / 'freshflow_ai' / 'requirements.txt')
        ])
        print("✅ Dependencies installed!")
    else:
        print("✅ All dependencies available")


def run_dashboard(port=8501):
    """Run the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / 'freshflow_dashboard.py'
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at {dashboard_path}")
        return
        
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🍃 FreshFlow AI - Inventory Decision Engine                ║
    ║                                                              ║
    ║   Starting dashboard on port {port}...                        ║
    ║   Open your browser to: http://localhost:{port}               ║
    ║                                                              ║
    ║   Press Ctrl+C to stop                                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run Streamlit
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        str(dashboard_path),
        '--server.port', str(port),
        '--server.headless', 'false',
        '--browser.gatherUsageStats', 'false'
    ])


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FreshFlow AI Dashboard')
    parser.add_argument('--port', type=int, default=8501, help='Port to run on')
    parser.add_argument('--check-only', action='store_true', help='Only check dependencies')
    
    args = parser.parse_args()
    
    print("\n🍃 FreshFlow AI - Inventory Decision Engine\n")
    print("=" * 50)
    
    check_dependencies()
    
    if args.check_only:
        print("\n✅ Dependency check complete")
        return
        
    print("\n" + "=" * 50)
    run_dashboard(args.port)


if __name__ == '__main__':
    main()
