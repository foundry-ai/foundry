import os
import rich
from rich.progress import Progress
from rich.markup import escape

from stanza.util.logging import logger
from pathlib import Path
import urllib
import asyncio
import threading
import functools

class Service:
    # Wait for all containers in the
    # service to finish
    async def wait():
        pass

    # Stop the service
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
    async def from_url(url):
        parsed = urllib.parse.urlparse(url)
        engine = Engine.from_name(parsed.scheme)
        return await engine.target(url)

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
        self.project_dir = os.path.abspath(project_dir)
        self.project_name = os.path.basename(self.project_dir)
        self.engine = PoetryEngine()

class PoetryProcessSet(Service):
    def __init__(self, procs):
        self._procs = procs
    
    async def wait(self):
        await asyncio.gather(*[p.wait() for p in self._procs])
    
    async def stop(self):
        for proc in self._procs:
            proc.terminate()
        await self.wait()
    

class PoetryLocal(Target):
    def __init__(self, n):
        self.engine = PoetryEngine()
        self.num_replicas = n

    async def _launch_proc(self, replica, image, args, env):
        if not isinstance(image, PoetryProject):
            raise RuntimeError("Can only launch PoetryProjects")
        env = dict(env)
        env['REPLICA'] = replica
        # forward the path so it can find poetry
        env['PATH'] = os.environ.get('PATH', '')
        env = {k:str(v) for k,v in env.items()}
        #cmd = " ".join(args)
        #logger.trace("poetry", f"Running [yellow]{escape(cmd)}[/yellow] in [yellow]{image.project_dir}[/yellow]")
        return await asyncio.create_subprocess_exec(
            *args,
            cwd=image.project_dir,
            # stdout=asyncio.subprocess.PIPE,
            # stderr=asyncio.subprocess.PIPE,
            env=env
        )

    async def launch(self, image, args, env={}):
        if not isinstance(image, PoetryProject):
            raise RuntimeError("Can only launch PoetryProjects")
        args = ["poetry", "run"] + list(args)
        return PoetryProcessSet(await asyncio.gather(*[
            self._launch_proc(i, image, args, env) \
                for i in range(self.num_replicas)]))

class PoetryEngine(Engine):
    async def ingest(self, src_image):
        if isinstance(src_image, PoetryProject):
            return src_image
        else:
            raise RuntimeError("Can't ingest other image types into PoetryEngine")

    async def target(self, url):
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        n = int(query.get('n', ['1'])[0])
        nodes = int(query.get('nodes', ['1'])[0])

        if parsed.netloc == 'localhost':
            return PoetryLocal(n)
        else:
            raise RuntimeError("Unrecognzied target")

register_engine("poetry", PoetryEngine)

# ------------------------- Docker-based container engine ------------------------

import docker

class DockerImage(Image):
    def __init__(self, engine, image_id):
        self.image_id = image_id
        self.engine = engine

class DockerContainers(Service):
    def __init__(self, engine, container_ids):
        self.engine = engine
        self.container_ids = container_ids
        self.ouput_thread = None
    
    def _forward_output(self):
        for o in self.engine.client.attach(self.container_id,
                        stdout=True, stderr=True,
                        logs=True, stream=True):
            line = o.decode('utf-8')
            print(line, end='')
    
    async def start(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.engine.client.start(self.container_id))
        self.output_thread = threading.Thread(target=self._forward_output)
        self.output_thread.start()
    
    async def stop(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.engine.client.stop(self.container_id))

class DockerLocal(Target):
    def __init__(self, engine, n):
        self.engine = engine
        self.num_replicas = n
    
    async def _launch(self, replica, image, args, env):
        container = await self.engine._create_local(replica, image, args, env)
        await container.start()
        return container

    # Launch all containers in parallel
    async def launch(self, image, args, env):
        containers = [self._launch(i, image, args, env) for i in range(self.num_replicas)]
        return await asyncio.gather(*containers)

class DockerService(Target):
    def __init__(self, engine, name, n=None):
        self.engine = engine
        self.name = name
        self.n = n

    async def launch(self, image, args, env):
        print('launching service', self.name, self.n)
        pass

class DockerEngine(Engine):
    def __init__(self, registry=None):
        self.client = docker.APIClient(base_url="unix:///var/run/docker.sock")

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
        return image_hash

    def _register_image(self, registry, image_tag):
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
            # If we never got a layer push event
            # just show a full progressbar anyways
            if not tasks:
                task = prog.add_task(" Pushed ")
                prog.update(task, total=1, completed=1)

        res = self.client.inspect_image(tag)
        image_digest = res["RepoDigests"][0] if res["RepoDigests"] else image_id

        logger.info(f"Pushed image [green]{image_tag}[/green] to {tag}")
        return image_digest

    def _create_local_blocking(self, replica, image, args, env={}):
        if not isinstance(image, DockerImage):
            raise RuntimeError("Can only launch docker images on docker target")
        args = ["poetry", "run"] + list(args)
        env = dict(env)
        env['DOCKER_IMAGE'] = image.image_id
        env['REPLICA'] = str(replica)
        if 'DOCKER_REGISTRY' in os.environ and not 'DOCKER_REGISTRY' in env:
            env['DOCKER_REGISTRY'] = os.environ['DOCKER_REGISTRY']
        if 'DOCKER_RUNTIME' in os.environ and not 'DOCKER_RUNTIME' in env:
            env['DOCKER_RUNTIME'] = os.environ['DOCKER_RUNTIME']
        runtime = os.environ.get("DOCKER_RUNTIME", "nvidia")
        container = self.client.create_container(
            image.image_id, args, tty=True, stdin_open=True,
            host_config=self.client.create_host_config(
                auto_remove=True,
                binds={
                    "/var/run/docker.sock": {
                        "bind": "/var/run/docker.sock",
                        "mode": "rw"
                    }
                },
                runtime=runtime
            ),
            runtime=runtime,
            environment=env
        )
        if not "Id" in container:
            raise RuntimeError("Failed to create container.", container)
        container_id = container["Id"]
        # get container info
        info = self.client.inspect_container(container_id)
        container = DockerContainer(self, container_id, replica)
        return container
    
    async def _create_local(self, replica, image, args, env={}):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._create_local_blocking(replica, image, args, env))
    
    async def ingest(self, image):
        if isinstance(image, DockerImage):
            return image
        elif isinstance(image, PoetryProject):
            image_id = self._build(image.project_name, image.project_dir)
            if 'DOCKER_REGISTRY' in os.environ:
                registry = os.environ['DOCKER_REGISTRY']
                image_id = self._register_image(registry, image.project_name)
            return DockerImage(self.client, image_id)

    async def target(self, url):
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.netloc != 'localhost':
            raise RuntimeError("Unrecognzied target")

        if parsed.path == '' or parsed.path == '/':
            n = int(query.get('n', ['1'])[0])
            return DockerLocal(self, n)
        elif parsed.path.startswith('/service/'):
            name = parsed.path[9:]
            n = int(query['n'][0]) if 'n' in query else None
            return DockerService(self, name, n)
        else:
            raise RuntimeError("Unrecognzied target")


register_engine("docker", DockerEngine)