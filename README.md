## A RL agent for multi-morphology control

### Setup

Versions used :
- Python 3.6.15 
- CUDA 12.3  
- MuJoCo 

Then you can install all the libraries using

```
pip install -r requirements.txt
```

### Running 

You can simply run the training procedure with the following command :

```
python main.py --expID [experience id] --td --bu --morphologies [morphology family]
```

Be carefull to check that the expID you run this experiment on is not already in the `results` or `buffers` folders. You must choose a morphology family to train your model on. It can be one of the following :
- hopper
- cheetah
- humanoid
- walker

For more information

### Code structure




### Acknowledgement

This project is done using previous work from these papers :
- [AnyMorph](https://arxiv.org/abs/2206.12279), with code in this [repo](https://github.com/montrealrobotics/AnyMorph)
- [Amorpheus](https://arxiv.org/abs/2010.01856), with code in this [repo](https://github.com/yobibyte/amorpheus)
- [Modular-rl](https://arxiv.org/abs/2007.04976), with code in this [repo](https://github.com/huangwl18/modular-rl)