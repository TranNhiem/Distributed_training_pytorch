"""
@TranNhiem 2022
About:
a script to build this environment on the nodes in our cluster, 
--> mount our registered dataset to these nodes, and submit a training job
Submit the multi-node multi-GPU PyTorch Lightning training script 
on an AzureML cluster according to the parameters in your file.

"""
import os
from Flag_configs.multi_gpus_training_config import read_cfg
from Flag_configs.absl_mock import Mock_Flag
read_cfg()
flag = Mock_Flag()
FLAGS = flag.FLAGS

## Implementation in Azure VM
## Using the MPI configureation
from azureml.core import (
    ScriptRunConfig,
    Workspace,
    Environment,
    Experiment,
    Dataset,
)

from azureml.core.runconfig import MpiConfiguration

def main():
    # Load run parameters
    with open("params.yml", "r") as stream:
        PARAMS = yaml.safe_load(stream)
    # Define Azure subscription and workspace.
    # You might need to replace os.environ["SUB_ID"]
    # with your Azure subscription ID or an
    # environment variable referencing this ID.
    ws = Workspace.get(
        name=FLAGS.ws_name,
        subscription_id=os.environ["SUB_ID"],
        resource_group=FLAGS.resource_group,
    )
    # Define Python environment from requirements file
    myenv = Environment.from_pip_requirements(
        name=FLAGS.exp_name + "_env",
        file_path="requirements.txt"
    )
    # Configure VMs in cluster with base image from Microsoft
    myenv.docker.enabled = True
    myenv.docker.base_image = (
        "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04"
    )
    # Attach to dataset created from Azureblobstore
    ds = Dataset.get_by_name(
        workspace=ws, name=FLAGS.dataset_name
    ).as_mount(path_on_compute=FLAGS.dataset_path)
    # Set distributed run parameters
    distributed_job_config = MpiConfiguration(
        node_count=FLAGS.nodes
    )
    # Define job to run on AML compute
    src = ScriptRunConfig(
        source_directory=".",
        script=FLAGS.job_script_name,
        compute_target=FLAGS.compute_target,
        distributed_job_config=distributed_job_config,
        environment=myenv,
        arguments=[ds],
    )
    # Define experiment
    experiment = Experiment(workspace=ws, name=FLAGS.exp_name)
    # Submit experiment run
    run = experiment.submit(config=src)
    run.wait_for_completion(show_output=True)

if __name__ == "__main__":
    main()