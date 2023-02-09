import argparse
import os
import sys
import docker
import dockerpty
import socket

import rich
from rich import print
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
    return image_hash, PROJECT_NAME

# Will launch a registry (if one is not running)
# and push the latest image hash to the registry
def register_image(registry, image_tag):
    # Tag appropriately for the push
    tag = f"{registry}/{image_tag}"

    # add another tag
    assert client.tag(image_tag, tag)

    # Show the push progress
    with Progress() as prog:
        tasks = {}
        res = client.push(tag, stream=True, decode=True)
        for r in res:
            p = r.get("progressDetail", {})
            total = p.get("total", None)
            current = p.get("current", None)

            task_id = r.get('id', None)
            if not task_id in tasks and total:
                tasks[task_id] = prog.add_task(" Pushing layer")
            if total:
                task = tasks[task_id]
                prog.update(task, total=total, completed=current)
            if "errorDetail" in r:
                m = r["errorDetail"]["message"]
                rich.print(f" [red]{m}[/red]")
                sys.exit(1)

    rich.print(f" Pushed image [green]{image_tag}[/green] to {tag}")
    return tag

def launch(image_id, image_tag, args):
    cmd = " ".join(args)
    rich.print(f"[yellow]-- Launching container[/yellow]")

    # get the full digest of the image being launched
    res = client.inspect_image(image_tag)
    # get the digest to be used
    # to deterministically pull/launch this particular image
    image_digest = res["RepoDigests"][0] if res["RepoDigests"] else image_id

    container = client.create_container(
        image_tag, args, tty=True, stdin_open=True,
        host_config=client.create_host_config(
            auto_remove=True,
            binds={
                "/var/run/docker.sock": {
                    "bind": "/var/run/docker.sock",
                    "mode": "rw"
                }
            }
        ),
        environment={
            "DOCKER_IMAGE": image_digest,
        },
    )
    if not "Id" in container:
        print("[red]Failed to create container.[/red]")
        print(container)
        sys.exit(1)

    container_id = container["Id"]
    # get container info
    info = client.inspect_container(container_id)
    name = info["Name"].lstrip("/")
    rich.print((f"-- Running [blue]{cmd}[/blue] in [green]{name}[/green]"
            f" (img: [yellow]{image_tag}[/yellow])"))
    client.start(container_id)
    outputs = client.attach(container_id, stream=True)
    dockerpty.start(client, container_id)

def run(args):
    # Launch the container
    image_id, tag = build_image()
    if args.registry:
        tag = register_image(args.registry, tag)
    launch(image_id, tag, [args.target] + args.target_args)

def main():
    parser = argparse.ArgumentParser(prog='launch')
    parser.add_argument("--registry",
        default=os.environ.get("DOCKER_REGISTRY", None))
    parser.add_argument("target")
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    run(args)

if __name__=="__main__":
    main()