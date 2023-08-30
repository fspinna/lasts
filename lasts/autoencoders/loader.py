from lasts.utils import get_project_root
from lasts.autoencoders.variational_autoencoder_v2 import load_model as load_vae_v2


def vae_list():
    root = get_project_root()
    path = (root / "autoencoders" / "cached" / "vae_v2").glob("**/*")
    return [x.name for x in path if x.is_dir()]


def vae_loader(name, verbose=False):
    root = get_project_root()
    path = root / "autoencoders" / "cached" / "vae_v2" / name / (name + "_vae")
    encoder, decoder, autoencoder = load_vae_v2(path, verbose=verbose)
    return encoder, decoder, autoencoder


if __name__ == "__main__":
    encoder, decoder, autoencoder = vae_loader("Coffee__ldim4")
