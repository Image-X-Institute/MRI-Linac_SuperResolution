import os


def configure_path_to_parameters(
    DOCKER: bool, DOCKERPATH: str, MRLPATH: str, model_name: str
) -> str:
    """
    Provides the right path depending on whether Gadgetron is being run
    for local/offline development (i.e., DOCKER=False) or online/real-time
    deployment (i.e., DOCKER=True). Set the DOCKER variable in load.py.

    :param DOCKER: is this running in the Docker container at the bunker?
    :param model_name: name of the JIT compiled model.
    """
    if DOCKER:
        path_to_parameters = os.path.join(DOCKERPATH, model_name)
    else:
        path_to_parameters = os.path.join(MRLPATH, model_name)

    return path_to_parameters
