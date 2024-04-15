all: test cov lint

test: 
	pytest

cov: 
	coverage run -m pytest

lint: 
	pylint app.py && flake8.py
