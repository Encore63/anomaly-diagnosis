import hydra

from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None,
            config_path="./configs",
            config_name="default")
def test(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    test()
