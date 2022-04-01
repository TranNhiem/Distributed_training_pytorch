from Flag_configs.absl_mock import Mock_Flag

def read_cfg(mod="non_contrastive"):
    flags = Mock_Flag()
    base_cfg()
    wandb_set()

def base_cfg():
    flags = Mock_Flag()

    flags.DEFINE_integer(
        'IMG_height', 224,
        'image height.')

    flags.DEFINE_integer(
        'IMG_width', 224,
        'image width.')

    flags.DEFINE_integer(
        'SEED', 26,  # 40, 26
        'random seed use for shuffle data Generate two same image ds_one & ds_two')

    flags.DEFINE_integer(
        'train_batch', 200,
        'Train batch_size .')

    flags.DEFINE_integer(
        'val_batch', 200,
        'Validaion_Batch_size.')

    flags.DEFINE_integer(
        'train_epochs', 100,
        'Number of epochs to train for.')

    flags.DEFINE_integer(
        'num_classes', 200,
        'Number of class in training data.')

    flags.DEFINE_string(
        'train_path', '/data1/share/1K_New/train/',
        'Train dataset path.')

    flags.DEFINE_string(
        'val_path', "/data1/share/1K_New/val/",
        'Validaion dataset path.')


    flags.DEFINE_string(
        'train_label', "image_net_1k_lable.txt",
        'train_label.')

    flags.DEFINE_string(
        'val_label', "ILSVRC2012_validation_ground_truth.txt",
        'val_label.')

def wandb_set():
    flags = Mock_Flag()
    flags.DEFINE_string(
        "wandb_project_name", "Distributed_multi_node_training",
        "set the project name for wandb."
    )
    flags.DEFINE_string(
        "wandb_run_name", "testing_multi_nodes_training",
        "set the run name for wandb."
    )
    flags.DEFINE_enum(
        'wandb_mod', 'run', ['run', 'dryrun'],
        'update the to the wandb server or not')

def machine_config(): 
    flags = Mock_Flag()

    flags.DEFINE_integer(
        "nodes",  2, 
        "Number of nodes in cluster training"
    )

    flags.DEFINE_integer(
        "num_gpus",  8, 
        "Number of nodes in cluster training"
    )

    flags.DEFINE_enum(
        "your_choice" , "op1", ["op1","op2"], 
        "your options"
    )

    flags.DEFINE_string(
        'resource_group', "<workspace-name>-common",
        'Replace with your resource group by providing the cluster name')
    
    flags.DEFINE_string(
        'compute_target', '<cluster-name>',
        'Replace with the name of your cluster')

    flags.DEFINE_string(
        'job_script_name', 'train.py',
        'The script to train the model')



