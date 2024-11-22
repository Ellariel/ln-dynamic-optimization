## Hybrid pathfinding optimization for the Lightning Network with Reinforcement Learning

### APA

```python
Valko, D., & Kudenko, D. (2024). Hybrid pathfinding optimization for the Lightning Network with Reinforcement Learning
```

### BibTeX
```python
@misc{ValkoKudenko2024,
title={ybrid pathfinding optimization for the Lightning Network with Reinforcement Learning}, 
author={Danila Valko and Daniel Kudenko},
year={2024},
publisher={},
howpublished={},
doi = {},
}
```

### Setup
It requires some virtual environment with certain dependencies, see `requirements.txt`.
```sh
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt 
```

### Run
* run training
```sh
source activate myenv && python train.py
```
* run experiments
```sh
source activate myenv && python test.py
```

## Sources and References

- Native pathfinding algorithms are based on [[Kumble & Roos, 2021]](https://ieeexplore.ieee.org/document/9566199); [[Kumble, Epema & Roos, 2021]](https://arxiv.org/pdf/2107.10070.pdf); see also, [GitHub](https://github.com/SatwikPrabhu/Attacking-Lightning-s-anonymity).
- Experiments were run on a snapshot of the Lightning Network obtained from [(Decker, 2020)](https://github.com/lnresearch/topology).


