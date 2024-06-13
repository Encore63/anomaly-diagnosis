import hydra

from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None,
            config_path="./configs/hydra_configs",
            config_name="default")
def test(cfg: DictConfig):
    data = 0
    config_yaml = OmegaConf.to_yaml(cfg)
    print(config_yaml)


if __name__ == '__main__':
    test()
