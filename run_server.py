#!/usr/bin/env python3
"""
Professional Loan Prediction Server Launcher
Provides flexible server configuration options.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Launch the loan prediction server with configurable options."""
    parser = argparse.ArgumentParser(
        description='Launch the Professional Loan Prediction Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server.py                    # Run on default port 5000
  python run_server.py --port 8080       # Run on port 8080
  python run_server.py --host 0.0.0.0    # Run on all interfaces (network access)
  python run_server.py --debug           # Run in debug mode
  python run_server.py --port 3000 --debug  # Custom port with debug
        """
    )

    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Port to run the server on (default: 5000)'
    )

    parser.add_argument(
        '--host', '-H',
        type=str,
        default='127.0.0.1',
        help='Host to bind the server to (default: 127.0.0.1 for localhost only)'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Run in debug mode with auto-reload'
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ['PORT'] = str(args.port)
    os.environ['HOST'] = args.host
    os.environ['FLASK_DEBUG'] = 'true' if args.debug else 'false'

    # Change to backend directory
    backend_dir = Path(__file__).parent / 'backend'
    os.chdir(backend_dir)

    # Add backend to Python path
    sys.path.insert(0, str(backend_dir))

    print(f"🚀 Starting Loan Prediction Server...")
    print(f"📍 Host: {args.host}")
    print(f"🌐 Port: {args.port}")
    print(f"🔧 Debug: {'Enabled' if args.debug else 'Disabled'}")
    print(f"🎯 URL: http://{args.host}:{args.port}")
    print("-" * 50)

    # Import and run the app
    try:
        from app import app, initialize_app

        if initialize_app():
            app.run(
                host=args.host,
                port=args.port,
                debug=args.debug,
                threaded=True
            )
        else:
            print("❌ Failed to initialize application")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()