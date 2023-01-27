import os
import pickle
from dataclasses import dataclass
from ode.logging import logger

@dataclass
class EmptyConfig:
    pass

# An acivity reports its data to an experiment
class Activity:
    def __init__(self, name, config_dataclass=None):
        self.name = name
        self.config_dataclass = config_dataclass
    
    def run(self, config, experiment):
        pass

class FuncActivity(Activity):
    def __init__(self, name, config_dataclass, exec):
        super().__init__(name, config_dataclass)
        self._exec = exec
    
    def run(self, config, *args, **kwargs):
        return self._exec(config, *args, **kwargs)

    def __call__(self, config, *args, **kwargs):
        return self._exec(config, *args, **kwargs)

# A decorator version for convenience
def activity(name, config_dataclass=None):
    def build_wrapper(f):
        return FuncActivity(name, config_dataclass, f)
    return build_wrapper

def launch_from(activities, analyses, args, root_dir, lab=None):
    import argparse
    from ode.experiment.config import parsable
    parser = argparse.ArgumentParser(
        prog='Experiment Launcher',
    )
    subparsers = parser.add_subparsers(title='commands')
    subparsers.dest = 'command'

    # Add the experiments
    exp_parser = subparsers.add_parser('exp')
    exp_subparsers = exp_parser.add_subparsers(title='experiments')
    exp_subparsers.dest = 'activity'

    act_map = {}
    for a in activities:
        exp_parser = exp_subparsers.add_parser(a.name)
        exp_parser.add_argument("--repo", default='dummy')
        config_class = parsable(a.config_dataclass)
        config_class.add_to_parser(exp_parser)
        act_map[a.name] = (a, config_class)
    
    # Add the analyses
    anl_parser = subparsers.add_parser('anl')
    anl_subparsers = anl_parser.add_subparsers(title='analyses')
    anl_subparsers.dest = 'analysis'

    anl_map = {}
    for a in analyses:
        anl_parser = anl_subparsers.add_parser(a.name)
        #anl_parser.add_argument("results", nargs="+", default=[], required=False)
        config_class = parsable(a.config_dataclass)
        config_class.add_to_parser(anl_parser)
        anl_map[a.name] = (a, config_class)

    args = parser.parse_args(args)

    if args.command == 'exp':
        if args.activity not in act_map:
            logger.error('Unrecognized acitvity')
            return
        activity, config_class = act_map[args.activity]
        config = config_class.from_args(args)

        from ode.experiment import Repo
        repo = Repo.from_url(args.repo)
        exp = repo.experiment(args.activity)
        run = exp.create_run()
        result = activity.run(config, run)
        if result is not None:
            os.makedirs(os.path.join(root_dir, 'results'), exist_ok=True)
            path = os.path.join(root_dir, 'results', f'{args.activity}.pkl')
            with open(path, 'wb') as file:
                pickle.dump(result, file)
        logger.info(f"Done with {args.activity}")
    elif args.command == 'anl':
        if args.analysis not in anl_map:
            logger.error('Unrecognized analysis')
            return
        data = []
        for r in []:
            path = os.path.join(root_dir, 'results', f'{r}.pkl')
            with open(path, 'rb') as file:
                data.append(pickle.load(file))
        activity, config_class = anl_map[args.analysis]
        config = config_class.from_args(args)
        result = activity.run(config, data)
        if result is not None:
            # result should contain a dictionary of figures
            if not isinstance(result, dict):
                result = {activity.name: result}
            os.makedirs(os.path.join(root_dir, 'figures'), exist_ok=True)
            for n, p in result.items():
                html_path = os.path.join(root_dir, 'results', f'{n}.html')
                png_path = os.path.join(root_dir, 'results', f'{n}.png')
                p.write_html(html_path)
                p.write_image(png_path)
    else:
        logger.error('Unrecognized command')