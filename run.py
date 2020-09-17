import argparse 
from warpmeta.config.config_loader import initialize_config, get_config
from warpmeta.data.data_loader import load_benchmark
from warpmeta.train import train
import wandb
from os import path

parser = argparse.ArgumentParser(description='Geometric Meta Learning')
parser.add_argument('-u', '--config_updates', type=str, default="", help=\
                    "Extra configuration to inject into config dict")
parser.add_argument('-f', '--config_file', type=str, default="config.json", help=\
                    "File to load config from")

def run(args):

    # Initialize Config
    initialize_config(args.config_file, load_config=
                      path.exists(args.config_file),
                      save_config=not path.exists(args.config_file),
                      config_updates=args.config_updates)

    config = get_config()
    # Initialize wandb dashboard
    wandb.init(project="geometric-meta-learning",
               config=config.as_json(),
               group=config.wandb_group)

    benchmark = load_benchmark(config.dataset, config.model)
    train(benchmark)




parser.set_defaults(func=run)

if __name__ == "__main__":
    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)
    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
