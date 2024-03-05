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
  title={SPARE: Shortest Path Global Item Relations for Efficient Session-based Recommendation},
  author={Peintner, Andreas and Mohammadi, Amir Reza and Zangerle, Eva},
  booktitle={Proceedings of the 17th ACM Conference on Recommender Systems},
  pages={58--69},
  year={2023}
}
```
