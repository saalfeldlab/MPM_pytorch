# MPM pytorch

pytorch version of https://github.com/yuanming-hu/taichi_mpm

Cédric Allier

Janelia Research Campus, Howard Hughes Medical Institute

![Simulation Demo](assets/cubes.gif)

```python
python -o generate 'multimaterial_1_3D.yaml'
```

![Simulation Demo](assets/cubes_F.gif)

```
data_generate(
    config,
    device=device,
    visualize=True,
    run_vizualized=0,
    style="black",  # Style options: "black", "latex", "F", "M"
                    # - "black": dark background
                    # - "latex": use LaTeX rendering
                    # - "F": color by deformation gradient magnitude
                    # - "M": color by material type
                    # - default (no "F" or "M"): color by particle ID
                    # Can combine: e.g., "black F" or "black latex M"
    alpha=1,
    erase=False,
    bSave=True,
    step=200,
)
```


### Setup
Run the following line from the terminal to create a new environment particle-graph:
```
conda env create -f environment.yaml
```

Activate the environment:
```
conda activate MPM-pytorch
```

Install the package by executing the following command from the root of this directory:
```
pip install -e .
```

Then, you should be able to import all the modules from the package in python:

```python
from MPM-pytorch import *
```
