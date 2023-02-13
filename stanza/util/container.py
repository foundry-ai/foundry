import os
import rich
from rich.progress import Progress
from rich.markup import escape

from stanza.logging import logger
from pathlib import Path
import urllib
import asyncio

class Container:
    # should have stdout, stderr, attributes
    # which should be of type asyncio.StreamReader

    # Wait for the container to finish
    async def wait():
        pass

    # Interrupt the container and wait until done
    async def stop():
        pass

class Image:
    # must have the "engine" attribute

    # gets the current sandbox in which
    # the project is running and returns the associated image
    # if there is no sandbox, it will return a PoetryProject object
    # which can be ingested into the engine of choice (usually determined by a target)
    @staticmethod
    def current():
        docker_image = os.environ.get("DOCKER_IMAGE", None)
        if docker_image:
            client = docker.APIClient(base_url="unix:///var/run/docker.sock")
            return DockerImage(client, docker_image)
        else:
            # try and find a folder with pyproject.toml in it
            root_dir = Path(os.getcwd()).absolute()
            while True:
                test_file = root_dir / "pyproject.toml"
                if test_file.is_file():
                    break
                else:
                    parent = root_dir.parent
                    if parent == root_dir:
                        raise ValueError("No project root found!")
                    root_dir = parent
            logger.trace("container", f"Using project root directory {root_dir}")
            return PoetryProject(str(root_dir))

# Images can be run ontop of targets.
# for instance poetry://local is a target which
# just runs the command in the local virtualenv
# docker://local is a target which runs in the local docker environment

class Target:
    # must have the "engine" attribute
    async def launch(self, image, cmd, env={}):
        pass

    @staticmethod
    def from_url(url):
        parsed = urllib.parse.urlparse(url)
        engine = Engine.from_name(parsed.scheme)
        return engine.target(url)

ENGINES = {}

def register_engine(name, engine):
    ENGINES[name] = engine

# The container engine
class Engine:
    # Will convert image into one compatible
    # with this engine.
    async def ingest(self, src_image):
        pass
    
    def target(self, url):
        pass
    
    @staticmethod
    def from_name(name):
        return ENGINES.get(name, None)()

# ----------------------- Poetry-based "container" engine ------------------------

class PoetryProject(Image):
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.engine = PoetryEngine()

class PoetryProcess(Container):
    def __init__(self, proc):
        self.proc = proc
    
    async def wait(self):
        await self.proc.wait()
    
    async def stop(self):
        self.proc.terminate()
        await self.proc.wait()
    
    @staticmethod
    async def launch(replica, image, args, env):
        if not isinstance(image, PoetryProject):
            raise RuntimeError("Can only launch PoetryProjects")
        env = dict(env)
        env['REPLICA'] = replica
        # forward the path so it can find poetry
        env['PATH'] = os.environ.get('PATH', '')
        env = {k:str(v) for k,v in env.items()}
        cmd = " ".join(args)
        #logger.trace("poetry", f"Running [yellow]{escape(cmd)}[/yellow] in [yellow]{image.project_dir}[/yellow]")
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=image.project_dir,
            env=env
        )
        return PoetryProcess(proc)

class PoetryLocal(Target):
    def __init__(self, n):
        self.engine = PoetryEngine()
        self.num_replicas = n
        
    async def launch(self, image, args, env={}):
        if not isinstance(image, PoetryProject):
            raise RuntimeError("Can only launch PoetryProjects")
        args = ["poetry", "run"] + list(args)
        return await asyncio.gather(*[PoetryProcess.launch(i, image, args, env) for i in range(self.num_replicas)])

class PoetryEngine(Engine):
    async def ingest(self, src_image):
        if isinstance(src_image, PoetryProject):
            return src_image
        else:
            raise RuntimeError("Can't ingest other image types into PoetryEngine")

    def target(self, url):
        parsed = urllib.parse.urlparse(url)

        query = urllib.parse.parse_qs(parsed.query)
        n = int(query.get('n', ['1'])[0])
        nodes = int(query.get('nodes', ['1'])[0])

        if parsed.netloc == 'localhost':
            return PoetryLocal(n)
        else:
            print(parsed)
            return PoetrySlurm(n)
        else:
            raise RuntimeError("Unrecognzied target")

register_engine("poetry", PoetryEngine)

# ------------------------- Docker-based container engine ------------------------

import docker

class DockerImage(Image):
    def __init__(self, engine, unique_ref):
        self.unique_ref = unique_ref
        self.engine = engine

class DockerLocal(Target):
    def __init__(self, engine):
        self.engine = engine
        self.num_replicas = 1

    async def launch(self, image, cmd, env={}, socket=False):
        cmd_str = " ".join(cmd)
        cmd = ["poetry", "run"] + args

        # make a copy so we don't modify original
        env = dict(env)
        env["DOCKER_IMAGE"] = self.image.unqiue_ref
        container = self.engine.client.create_container(
            self.image.unique_ref, args, tty=True, stdin_open=True,
            host_config=self.client.create_host_config(
                auto_remove=True,
                binds={
                    "/var/run/docker.sock": {
                        "bind": "/var/run/docker.sock",
                        "mode": "rw"
                    }
                }
            ),
            environment=env
        )
        if not "Id" in container:
            raise RuntimeError("Failed to create container.", container)
        container_id = container["Id"]
        # get container info
        info = self.engine.client.inspect_container(container_id)
        self.engine.client.start(container_id)

class DockerEngine(Engine):
    def __init__(self, registry=None):
        self.client = docker.APIClient(base_url="unix:///var/run/docker.sock")
        self.registry = registry or os.environ.get("DOCKER_REGISTRY", None)
    
    def _build(self, name, root_dir):
        logger.info(f"[yellow]Creating {name} image [/yellow]")

        image_hash = None
        # Logs only get printed
        # if the build does not complete
        logs = []

        with Progress() as prog:
            task = prog.add_task(" Building Image")

            output = self.client.build(path=root_dir, tag=name,
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
            logger.error("[red]Error building image:[/red]")
            for l in logs:
                print(l)
            raise RuntimeError("Unable to build container!")
        logger.info(f"Built ID: [blue]{escape(image_hash)}[/blue]")
        return image_hash, name

    def _register_image(self, image_tag):
        # Tag appropriately for the push
        tag = f"{registry}/{image_tag}"

        # add another tag
        assert self.client.tag(image_tag, tag)

        # Show the push progress
        with Progress() as prog:
            tasks = {}
            res = self.client.push(tag, stream=True, decode=True)
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
                    raise RuntimeError("Unable to upload container!")

        res = self.client.inspect_image(tag)
        image_digest = res["RepoDigests"][0] if res["RepoDigests"] else image_id

        rich.print(f" Pushed image [green]{image_tag}[/green] to {tag}")
        return image_digest
    
    def build(self, image_name, ctx_dir):
        image_id, image_tag = self._build(image_name, ctx_dir)
        if self.registry:
            image_digest = self._register_image(image_tag)
        else:
            image_digest = image_id
        return DockerImage(self.client, image_digest)

    def target(self, url):
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc == 'localhost':
            return DockerLocal()
        else:
            raise RuntimeError("Unrecognzied target")