import argparse
import os
import sys
import docker
import socket

import rich
from rich.progress import Progress
from rich.markup import escape

ROOT_DIR = os.path.abspath(os.path.join(__file__, '..', '..'))
PROJECT_NAME = os.path.basename(ROOT_DIR)
HOSTNAME = socket.gethostname()

PREFIX_ARGS = ["poetry", "run"]

client = docker.APIClient(base_url="unix:///var/run/docker.sock")

# Will ensure the docker registry container is running,
# deploy to the registry, 
# and return the image hash of the deployed image
def build_image():
    rich.print(f"[yellow]-- Creating {PROJECT_NAME} image [/yellow]")

    image_hash = None
    # Logs only get printed
    # if the build does not complete
    logs = []

    with Progress() as prog:
        task = prog.add_task(" Building Image")

        output = client.build(path=ROOT_DIR, tag=PROJECT_NAME,
                        rm=True, decode=True)
        for o in output:
            if 'stream' in o:
                line = o['stream'].strip()
                if line:
                    logs.append(line)
                    # parse the percentage done
                    if line.startswith('Step '):
                        c, t = line.split(' ')[1].split('/')
                        c = int(c)
                        t = int(t)
                        prog.update(task, total=t, completed=c)
            elif 'aux' in o:
                image_hash = o['aux']['ID']
    if not image_hash:
        rich.print('[red]Error building image:[/red]')
        for l in logs:
            print(l)
    rich.print(f' Built ID: [blue]{escape(image_hash)}[/blue]')
    return image_hash

# Will launch a registry (if one is not running)
# and push the latest image hash to the registry
def register_image(registry, image_hash):
    # Tag appropriately for the push
    tag = f"{registry}/{PROJECT_NAME}"
    assert client.tag(PROJECT_NAME, tag)
    with Progress() as prog:
        task = prog.add_task(" Pushing Image")
        res = client.push(tag, stream=True, decode=True)
        for r in res:
            p = r.get("progressDetail", {})
            total = p.get("total", None)
            current = p.get("current", None)
            if total:
                prog.update(task, total=total, completed=current)
            if "errorDetail" in r:
                m = r["errorDetail"]["message"]
                rich.print(f" [red]{m}[/red]")
                sys.exit(1)

    rich.print(f" Pushed image [green]{PROJECT_NAME}[/green] to {tag}")
    return tag

def launch(image_tag, args):
    cmd = " ".join(args)
    rich.print(f"[yellow]-- Launching container[/yellow]")
    rich.print(f"-- Running [blue]{cmd}[/blue] in [green]{image_tag}[/green]")

def run(args):
    # Launch the container
    image_hash = build_image()
    tag = register_image(args.registry, image_hash)
    launch(tag, [args.target] + args.target_args)

def main():
    parser = argparse.ArgumentParser(prog='launch')
    parser.add_argument("--registry", default=f'kronos:5000')
    parser.add_argument("target")
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    run(args)

if __name__=="__main__":
    main()