# Towards Interpretable Graph Classification
## Getting Started
To obtain heatmaps:
```
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py cuda=0 -gm=DGCNN -data=MUTAG
```

To calculate metrics:
```
python3 calculate_metrics.py cuda=0 -gm=DGCNN -data=MUTAG
```