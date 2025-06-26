"""
Advanced Auto-Launch Script for Emotion Detection App
Includes automatic setup, dependency management, and error handling
Author: DevanshSrajput
Date: 2025-06-26
"""

import os
import sys
import subprocess
import time
import platform
import socket
import webbrowser
import signal
from threading import Timer

class EmotionAppLauncher:
    """Class to handle launching the Emotion Detection Streamlit app"""
    
    def __init__(self):
        self.app_name = "🎭 Emotion Detection App"
        self.app_file = "streamlit_app.py"
        self.port = 8501
        self.browser_opened = False  # Track if browser was already opened
        self.process = None  # Store the streamlit process
        self.required_files = [
            'streamlit_app.py',
            'emotion_detector.py', 
            'visualizations.py'
        ]
        self.required_packages = [
            'streamlit==1.46.0',
            'pandas==2.0.3',
            'numpy==1.24.3',
            'scikit-learn==1.3.0',
            'nltk==3.8.1',
            'matplotlib==3.7.2',
            'seaborn==0.12.2',
            'plotly==5.15.0',
            'wordcloud==1.9.2',
            'joblib==1.3.1'
        ]
    
    def print_header(self):
        """Print application header"""
        print("🎭 EMOTION DETECTION AUTO-LAUNCHER")
        print("=" * 50)
        print(f"👤 User: DevanshSrajput")
        print(f"📅 Date: 2025-06-26 11:25:27")
        print(f"💻 Platform: {platform.system()} {platform.release()}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print("=" * 50)
    
    def check_files(self):
        """Check if all required files exist"""
        print("\n📁 Checking required files...")
        missing_files = []
        
        for file in self.required_files:
            if os.path.exists(file):
                print(f"✅ {file} found")
            else:
                print(f"❌ {file} missing")
                missing_files.append(file)
        
        if missing_files:
            print(f"\n❌ Missing files: {missing_files}")
            print("Please ensure all required files are in the current directory.")
            return False
        
        print("✅ All required files found!")
        return True
    
    def install_packages(self):
        """Install required packages"""
        print("\n📦 Installing/Updating packages...")
        
        for package in self.required_packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️ Warning: Could not install {package}")
    
    def setup_nltk(self):
        """Download NLTK data"""
        print("\n📚 Setting up NLTK data...")
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            print("✅ NLTK data downloaded")
        except Exception as e:
            print(f"⚠️ Warning: Could not download NLTK data: {e}")
    
    def check_port(self):
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', self.port))
                if result == 0:
                    print(f"⚠️ Port {self.port} is already in use")
                    print("❓ Another instance might be running. Continue anyway? (y/n): ", end="")
                    response = input().lower()
                    return response.startswith('y')
                return True
        except Exception as e:
            print(f"⚠️ Could not check port: {e}")
            return True
    
    def open_browser_delayed(self):
        """Open browser after delay - only once"""
        def open_browser():
            if not self.browser_opened:
                try:
                    # Wait a bit longer to ensure Streamlit is fully loaded
                    time.sleep(2)
                    webbrowser.open(f'http://localhost:{self.port}')
                    self.browser_opened = True
                    print(f"🌐 Browser opened at http://localhost:{self.port}")
                except Exception as e:
                    print(f"⚠️ Could not open browser: {e}")
                    print(f"Please manually open: http://localhost:{self.port}")
        
        # Wait 5 seconds for streamlit to start
        Timer(5.0, open_browser).start()
    
    def create_config(self):
        """Create Streamlit config to prevent auto-browser opening"""
        config_dir = os.path.expanduser('~/.streamlit')
        config_file = os.path.join(config_dir, 'config.toml')
        
        config_content = f"""
[server]
headless = false
port = {self.port}
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = {self.port}

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true

[logger]
level = "error"
messageFormat = "%(asctime)s %(message)s"

[client]
showErrorDetails = false
toolbarMode = "minimal"
"""
        
        try:
            os.makedirs(config_dir, exist_ok=True)
            with open(config_file, 'w') as f:
                f.write(config_content)
            print("✅ Streamlit config created")
        except Exception as e:
            print(f"⚠️ Could not create config: {e}")
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C signal"""
        print("\n\n🛑 Shutting down...")
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        print("✨ Thank you for using Emotion Detection App!")
        sys.exit(0)
    
    def launch_app(self):
        """Launch the Streamlit application"""
        print(f"\n🚀 Launching {self.app_name}...")
        print("-" * 40)
        print(f"📱 App will open at: http://localhost:{self.port}")
        print("🔄 Loading may take a few seconds...")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 40)
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Open browser automatically (only once) with delay
        self.open_browser_delayed()
        
        try:
            # Launch streamlit with specific config to prevent auto-browser opening
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                self.app_file,
                "--server.port", str(self.port),
                "--server.headless", "false",
                "--browser.gatherUsageStats", "false",
                "--server.address", "localhost",
                "--global.developmentMode", "false"
            ]
            
            # Set environment variable to prevent auto browser opening
            env = os.environ.copy()
            env['STREAMLIT_SERVER_HEADLESS'] = 'true'
            env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
            
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
            # Monitor the process and filter output
            print("\n📊 Streamlit Status:")
            print("-" * 30)
            
            if self.process.stdout is not None:
                for line in iter(self.process.stdout.readline, ''):
                    if line:
                        line = line.rstrip()
                        # Filter out browser-related messages and show only important ones
                        if any(keyword in line.lower() for keyword in [
                            'local url', 'network url', 'running on', 
                            'error', 'exception', 'traceback'
                        ]):
                            # Don't show URLs since we handle browser opening
                            if 'local url' not in line.lower() and 'network url' not in line.lower():
                                print(line)
                        elif 'you can now view your streamlit app' in line.lower():
                            print("✅ Streamlit app is running successfully!")
            else:
                print("⚠️ Unable to capture Streamlit output.")
            
            self.process.wait()
            
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
            self.signal_handler(signal.SIGINT, None)
        except FileNotFoundError:
            print(f"❌ Could not find {self.app_file}")
            print("Please ensure streamlit_app.py exists in the current directory")
        except Exception as e:
            print(f"❌ Error launching app: {e}")
    
    def run(self):
        """Main run method"""
        self.print_header()
        
        # Step 1: Check files
        if not self.check_files():
            input("\nPress Enter to exit...")
            return
        
        # Step 2: Install packages
        print("\n🔄 Setting up environment...")
        self.install_packages()
        
        # Step 3: Setup NLTK
        self.setup_nltk()
        
        # Step 4: Create config
        self.create_config()
        
        # Step 5: Check port
        if not self.check_port():
            input("\nPress Enter to exit...")
            return
        
        # Step 6: Launch app
        print("\n✨ Setup complete! Launching app...")
        time.sleep(1)
        self.launch_app()

def main():
    """Main function"""
    launcher = EmotionAppLauncher()
    launcher.run()

if __name__ == "__main__":
    main()