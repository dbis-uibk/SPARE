# SPARE

SPARE: Shortest Path Global Item Relations for Efficient Session-based Recommendation.

Use `pipenv` to create virtual environment. 

Preprocessed datasets already in `datasets` folder. 

## Usage
Tmall dataset as example:
- Construct global item graph, run:
    ```
    python build_global_graph.py --dataset tmall
    ```

- Sample positive and negative sessions, run:
     ```
    python pos_neg_sessions.py --dataset tmall
    ```

- Train GNN, run:
     ```
    python main.py --dataset tmall
    ```

## Requirements
See `Pipfile`.

## Citation
TBA