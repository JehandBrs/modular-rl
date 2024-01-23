'''
Here is the dictionary that maps a morphology name into its limbs indexes.
'''

# Function to automatically build the morphology sequence to mapping 
def build_name_to_morphology_sequence_mapping(envs_train, envs_train_names):
    """Function to create from a lots of environments the appropriate embedding from morphology to embedding"""
    
    # Create a mapping dictionary from morphology base to    
    name_prefix_to_motor_mapping = {}
    
    # We go through all morphologies and make a map of all possible limbs
    for env_train, env_train_name in zip(envs_train, envs_train_names):
        # Get name preifx and motors of the environment
        name_prefix = env_train_name.split('_')[0]
        motors = env_train().motors
        # Add it to the dictionary if not in yet
        if env_train_name.split('_')[0] not in name_prefix_to_motor_mapping.keys():
            name_prefix_to_motor_mapping[name_prefix]=[]
        # Add the motor name in 
        for motor in motors:
            if motor not in name_prefix_to_motor_mapping[name_prefix]:
                name_prefix_to_motor_mapping[name_prefix].append(motor)
                
    # We then map, for every env_name_prefix, each limb to a unique integer
    name_prefix_to_motor_indexes_mapping = {
        name_pref+'_'+motor_name : idx for name_pref, motors_names in name_prefix_to_motor_mapping.items() for idx, motor_name in enumerate(motors_names)
    }
    
    # Finally, we et the unique index for all morphology
    name_to_morphology_sequence_mapping_return = {}
    for env_train, env_train_name in zip(envs_train, envs_train_names): 
        motors = env_train().motors   
        name_pref = env_train_name.split('_')[0]
        
        name_to_morphology_sequence_mapping_return[env_train_name] = [
            name_prefix_to_motor_indexes_mapping[name_pref+'_'+motor] for motor in motors
        ]
        
    return name_to_morphology_sequence_mapping_return

# Manually built morphology to sequence mapping
name_to_morphology_sequence_mapping = {
    'cheetah_7_full': [0,1,2,3,4,5],
    'cheetah_6_front': [0,1,2,4,5],
    'cheetah_6_back': [1,2,3,4,5],
    'cheetah_5_balanced': [1,2,4,5],
    'cheetah_5_front': [0,1,2,5],
    'cheetah_5_back': [2,3,4,5],
    'cheetah_4_front': [1,2,5],
    'cheetah_4_back': [2,4,5],
    'cheetah_4_allfront': [0,1,2],
    'cheetah_4_allback': [3,4,5],
    'cheetah_3_front': [1,2],
    'cheetah_3_back': [4,5],
    'cheetah_3_balanced': [2,5],
    'cheetah_2_front': [2],
    'cheetah_2_back': [5],
####################################
    'walker_7_main': [0,1,2,3,4,5],
    'walker_7_flipped': [0,1,2,3,4,5],
    'walker_6_main': [1,2,3,4,5],
    'walker_6_flipped': [0,1,2,4,5],
    'walker_5_main': [1,2,4,5],
    'walker_5_flipped': [1,2,4,5],
    'walker_4_main': [2,4,5],
    'walker_4_flipped': [1,2,5],
    'walker_3_main': [2,5],
    'walker_3_flipped': [2,5],
    'walker_2_main': [2],
    'walker_2_flipped': [5],
####################################
    'humanoid_2d_9_full': [0,1,2,3,4,5,6,7],
    'humanoid_2d_8_right_knee': [0,1,2,4,5,6,7],
    'humanoid_2d_8_left_knee': [0,1,2,3,4,5,6],
    'humanoid_2d_7_right_leg': [0,1,4,5,6,7],
    'humanoid_2d_7_left_leg': [0,1,2,3,4,5],
    'humanoid_2d_7_right_arm': [2,3,4,5,6,7],
    'humanoid_2d_7_lower_arms': [0,2,3,4,6,7],
    'humanoid_2d_7_left_arm': [0,1,2,3,6,7],
####################################
    'hopper_3': [0, 3],
    'hopper_4': [0, 1, 3],
    'hopper_5': [0, 1, 2, 3],
}