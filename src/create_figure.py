import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf
from arguments import get_args_figure
from utils import load_dict_from_file
from test_train_morphologies import train_morphologies
import os
import json


sns.set_theme()


if __name__ == "__main__":
    
    # Args
    args = get_args_figure()
    
    # Path to the TensorFlow event file    
    results_directory = f'./{args.results_directory}'
    experiment_folder_name = f'/EXP_{args.ExpID}'
    experiment_directory = results_directory+experiment_folder_name
    for filename in os.listdir(experiment_directory):
        if args.logs_file_name_base in filename: 
            event_file_name = f'/{filename}'
    experiment_file_path = experiment_directory+event_file_name
    
    # Retrieve the arguments of the raining procedure
    file_path = experiment_directory+'/args.txt'
    training_args = load_dict_from_file(file_path)
        
        
    # Retrieve the data from the tensorboard data file
    steps, tags, values = [], [], []
    
    for i, e in enumerate(tf.compat.v1.train.summary_iterator(experiment_file_path)):
        for v in e.summary.value:
            if 'episode_reward' in v.tag:
                steps.append(e.step)
                tags.append(v.tag)
                values.append(v.simple_value)
    
    # Store into dataframe
    df = pd.DataFrame.from_dict({'step': steps, 'tag': tags, 'value': values})
    df = df[df['tag'].apply(lambda x: x.replace('_episode_reward', '') in train_morphologies)]
    df_rolling_mean = df.groupby('step', as_index=False).agg({'value':'mean'}).rolling(100, min_periods=0).mean()
    name_morphology_family = tags[0].split('_')[0]+'s'

    # Create the figure
    plt.figure(figsize=(10, 6))  # You can adjust the figure size
    sns.lineplot(data=df, x = 'step', y = 'value', estimator = 'mean', ci = None, alpha = 0.2)
    sns.lineplot(data=df_rolling_mean, x = 'step', y = 'value')
    plt.xlabel('Training steps')
    plt.ylabel('Training reward')
    plt.suptitle(f'{name_morphology_family}', fontsize=25)
    try:
        plt.title(f'With actor : {training_args["actor_type"]} | critic : {training_args["critic_type"]}', fontsize=10)
    except:
        do_nothing = None
        
    # Path to the images folders
    path_to_images_folder = './images/'
    image_name = f'{name_morphology_family}_{args.ExpID}_test.png'
    
    # Save the figure
    plt.savefig(path_to_images_folder+image_name) 

