'''
Here are lists wioth the morphologies we train our models on, 
and the morphologies we use to test the zero-shot performance of our models
'''

# The morphologies we train on :

train_morphologies = [
    'cheetah_2_back',
    'cheetah_2_front',
    'cheetah_3_back',
    'cheetah_3_front',
    'cheetah_4_allback',
    'cheetah_4_allfront',
    'cheetah_4_back',
    'cheetah_4_front',
    'cheetah_5_balanced',
    'cheetah_5_front',
    'cheetah_6_back',
    'cheetah_7_full',
    'walker_2_main',
    'walker_4_main',
    'walker_5_main',
    'walker_7_main',
    'humanoid_2d_7_left_arm',
    'humanoid_2d_7_lower_arms',
    'humanoid_2d_7_right_arm',
    'humanoid_2d_7_right_leg',
    'humanoid_2d_8_left_knee',
    'humanoid_2d_9_full',
    'hopper_3',
    'hopper_4',
    'hopper_5'
]

# The held-out morphologies used to test the zero shot performances of our model :

test_morphologies = [
    'cheetah_3_balanced', 
    'cheetah_5_back',
    'cheetah_6_front',
    'walker_3_main', 
    'walker_6_main',
    'humanoid_2d_7_left_leg', 
    'humanoid_2d_8_right_knee',
]