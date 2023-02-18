# Fashion-MNIST CNN

A CNN for classifying [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

## To Run

	python3 -m venv .venv
	source .venv/bin/activate
    pip install -r requirements.txt
    python main.py

## MacOS Setup Instructions

On an M2 Mac running MacOS 13 I was not able to install tensorflow using the suggested `pip install tensorflow` workflow.
Instead I had to run the following commands:

	brew install python@3.8
	/opt/homebrew/Cellar/python@3.8/3.8.16/bin/python3.8  -m venv .venv
	source .venv/bin/activate # I have this aliased to a command called 'activate'
	pip install --upgrade pip
	pip install tensorflow-macos
