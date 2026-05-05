PYTHON = python

run:
	$(PYTHON) main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

install:
	pip install numpy matplotlib

format:
	black *.py