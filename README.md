# SPARE

SPARE: Shortest Path Global Item Relations for Efficient Session-based Recommendation.

Preprocessed datasets already in `datasets` folder. 

## Usage
Tmall dataset as example:
- Construct global item graph, run:
    ```
    python build_graph.py --dataset tmall
    ```

- Train GNN, run:
     ```
    python main.py --dataset tmall
    ```

## Requirements
- Python 3
- PyTorch
- nltk
- scipy
- tqdm

## Citation
```
@inproceedings{peintner2023spare,
  author       = {Andreas Peintner and
                  Amir Reza Mohammadi and
                  Eva Zangerle},
  title        = {{SPARE:} Shortest Path Global Item Relations for Efficient Session-based
                  Recommendation},
  booktitle    = {Proceedings of the 17th {ACM} Conference on Recommender Systems, RecSys 2023},
  pages        = {58--69},
  publisher    = {{ACM}},
  year         = {2023},
}
```
