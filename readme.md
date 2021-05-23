# MBPO PyTorch
A PyTorch reimplementation of MBPO (When to trust your model: model-based policy optimization)

# Dependency

Please refer to ./requirements.txt.

# Usage

    pip install -e .

    # default hyperparams in ./configs/mbpo.yaml
    # remember to CHANGE proj_dir to your actual directory 
    python ./mbpo_pytorch/scripts/run_mbpo.py
    
    # you can also overwrite hyperparams by passing args, e.g.
    python ./mbpo_pytorch/scripts/run_mbpo.py --set seed=0 verbose=1 device="'cuda:0'" env.env_name='FixedHopper'

```
conda activate mbpo_pytorch
cd thl/causal_mbpo_torch
python ./mbpo_pytorch/scripts/run_mbpo.py --configs "mbpo.yaml" "halfcheetah.yaml" "priv.yaml"
python ./mbpo_pytorch/scripts/run_mbpo.py --configs "mbpo.yaml" "walker2d.yaml" "priv.yaml"
python ./mbpo_pytorch/scripts/run_mbpo.py --configs "mbpo.yaml" "hopper.yaml" "priv.yaml"
python ./mbpo_pytorch/scripts/run_mbpo.py --configs "mbpo.yaml" "ant.yaml" "priv.yaml"
``
  
# Credits
1. [vitchyr/rlkit](https://github.com/vitchyr/rlkit)
2. [JannerM/mbpo](https://github.com/JannerM/mbpo)
3. [WilsonWangTHU/mbbl](https://github.com/WilsonWangTHU/mbbl)