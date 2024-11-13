# Query reformulation

The report for this project is in [Report](./report.md)

The demo of project is [here](https://huggingface.co/spaces/CelDom/querysuggestions). 
Since the demo is run on very weak CPU. Therefore, the run time is much slower on the normal consumer CPU. Proof can be provided.

This repository contains a method that uses NLP to convert a user’s question into a search engine query, which can be used to find information to answer the question. It can also provide a refined version of the query.
## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

This system is designed to suggest possible queries based on a user’s question. The goal is to use these queries to search for information that can help answer the question.

## Installation

0. Make sure you install with python 3.12 to make the installation smooth and on Linux.

1. Clone the repository:
    ```bash
    git clone https://github.com/thhung/query_reform.git
    cd query_reform
    ```
2. Create virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use venv\Scripts\activate
    ```

3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Usage
4. Download the requirements files
    ```bash
    bash prepare.sh
    ```
5. Launch app or api
    ```bash
    streamlit run app.py # launch app
    bash run_api.sh # lauch api
    ```
6. To test api
    ```bash
    curl -X POST "http://localhost:8000/process" -H "Content-Type: application/json" -d '{"phrase": "What are the best ways to learn Python?"}'
    ```

## License
Check [License](./LICENSE)


