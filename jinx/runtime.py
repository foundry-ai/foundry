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
    
    def run(self, config, experiment):
        self._exec(config, experiment)

    def __call__(self, config, experiment, *args, **kwargs):
        self._exec(config, experiment, *args, **kwargs)

# A decorator version for convenience
def activity(name, config_dataclass=None):
    def build_wrapper(f):
        return FuncActivity(name, config_dataclass, f)
    return build_wrapper

def launch_from(activities, analyses, args, root_dir, lab=None):
    import argparse
    from jinx.experiment.config import parsable

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
        config_class = parsable(a.config_dataclass)
        config_class.add_to_parser(exp_parser)
        act_map[a.name] = (a, config_class)
    
    # Add the analyses
    args = parser.parse_args(args)

    if args.command == 'exp':
        if lab is None:
            from .aim import AimLab
            lab = AimLab(root_dir)
        if args.activity not in act_map:
            logger.error('Unrecognized acitvity')
            return
        activity, config_class = act_map[args.activity]
        config = config_class.from_args(args)
        experiment = lab.create(args.activity)
        activity.run(config, experiment)
        experiment.finish()
    else:
        logger.error('Unrecognized command')
