from funshot.run import run
import sys

if __name__ == "__main__":
    configs = sys.argv[1:]
    for config in configs:
        run(config_name=config)
