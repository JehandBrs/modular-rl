## A RL agent for multi-morphology control

### Setup

#### Requirements
- Python-3.6.15 
- PyTorch-1.10.0
- CUDA-12.3  
- [MuJoCo-200](https://www.roboti.us/index.html): download binaries, put license file inside, and add path to .bashrc

#### Setting up repository
  ```Shell
  git clone https://github.com/JehandBrs/modular-rl.git
  cd modular-rl/
  python3.6 -m venv mrEnv
  source $PWD/mrEnv/bin/activate
  ```

#### Installing Dependencies
  ```Shell
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

### Running 

You can simply run the training procedure with the following command :

```
python main.py --expID [experience id] --td --bu --morphologies [morphology family(ies)]
```

You must choose a morphology family to train your model on. It can be one or multiple of the following :
- hopper
- cheetah
- humanoid
- walker

For more information about how to run `main.py`, go to the `README_modular_rl.md` file.
When this script is launched, the experiment is associated with an ID and the results (model, tensorboard, buffer) are stored in the `results` and `buffers` files.

### Code structure

The code is structured the following way :

- the files `ModularActor.py` and `ModularCritic.py` refer to the models of the paper modular-rl.
- the files `TransformerActor.py` and `TransformerCritic.py` refer to the models of the paper amorpheus.
- the files `decoder_base.py`, `VariationalActor.py` and `VariationalCritic.py` refer to the models of the paper AnyMorph.
- the files `AnyMorphArchitecture.py`, `TD3AnyMorph.py`, `MorphologyEncoder.py` and `NameToMoprhologyMapping.py` refers to an attempt to code AnyMorph model from scratch and are not used in the training procedure.

### Acknowledgement

This project is done using previous work from these papers :
- [AnyMorph](https://arxiv.org/abs/2206.12279), with code in this [repo](https://github.com/montrealrobotics/AnyMorph)
- [Amorpheus](https://arxiv.org/abs/2010.01856), with code in this [repo](https://github.com/yobibyte/amorpheus)
- [Modular-rl](https://arxiv.org/abs/2007.04976), with code in this [repo](https://github.com/huangwl18/modular-rl)