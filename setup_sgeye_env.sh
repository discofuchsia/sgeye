#!/bin/bash

echo "🚀 Setting up SGEYE Python environment with pyenv..."

# Ensure pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv not found. Please install pyenv with Homebrew: brew install pyenv"
    exit 1
fi

# Ensure pyenv-virtualenv is installed
if ! pyenv commands | grep -q "virtualenv"; then
    echo "❌ pyenv-virtualenv not found. Install it with: brew install pyenv-virtualenv"
    exit 1
fi

# Install Python 3.10.13 if not already
if ! pyenv versions | grep -q "3.10.13"; then
    echo "📦 Installing Python 3.10.13..."
    pyenv install 3.10.13
fi

# Create virtual environment
echo "🧪 Creating virtualenv 'sgeye-env'..."
pyenv virtualenv 3.10.13 sgeye-env

# Activate environment
echo "⚡ Activating environment..."
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
pyenv activate sgeye-env

# Set local Python version
pyenv local sgeye-env

# Install dependencies
echo "📦 Installing requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo "To activate it in the future, run: pyenv activate sgeye-env"
echo "To run your app: streamlit run ai2.py"
