# Towards Interpretable Graph Classification
## Getting Started
To obtain heatmaps:
```
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py cuda=0 -gm=DGCNN -data=MUTAG
```

## Subgraph analysis:
```
python3 subgraph_analysis.py cuda=0 -gm=DGCNN -data=MUTAG
```
Optional params:  
-`graphsig`: set to 1 to run graphsig subgraph analysis. Default is 0.  
-`subgraph_explainability`: set to 1 to run explainability method based subgraph analysis. Default is 0.