# A tinyrpc-rpyc hybrid
# supports both reference objects

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

@dataclass
class Request(ABC):
    unique_id : str
    method_name: str
    args: Any
    kwargs: Any
    
    @abstractmethod
    def respond(self, result):
        ...
    
    @abstractmethod
    def error_respond(self, result):
        ...

    @abstractmethod
    def serialize(self):
        ...

@dataclass
class Response(ABC):
    unique_id: str
    result: Any
    error: Any = None

    @abstractmethod
    def serialize(self) -> bytes:
        ...

class Protocol(ABC):
    @abstractmethod
    def create_request(self, method_name, args, kwargs):
        ...
    
    @abstractmethod
    def parse_request(self, bytes) -> Request:
        ...

    @abstractmethod
    def parse_response(self, bytes) -> Response:
        ...

# A tinyrpc-style RPC system, with asyncio support
class Dispatcher:
    def __init__(self):
        self._methods = {}
    
    def get_method(self, name):
        if name in self._methods:
            return self._methods[name]
        raise RPCError(f"Method '{name}' not found")

    def add_method(self, f, name=None):
        assert callable(f)
        if not name:
            name = f.__name__
        if name in self._methods:
            raise RPCError(f"Method '{name}' already registered")
        self._methods[name] = f
        
    def public(self, name=None):
        # @public is decorated directly
        if callable(name):
            self.add_method(name)
            return name
        def decorate(f):
            self.add_method(f, name=name)
            return f
        return decorate
    
    async def dispatch(self, request: RPCRequest) -> RPCResponse:
        method = self.get_method(request.method_name)
        try:
            result = await method(*request.args, **request.kwargs)
            response = request.respond(result)
        except KeyboardInterrupt as e:
            raise e
        except _ as e:
            response = request.respond_error(e)
        return response

class RPCError(RuntimeError):
    def __init__(self, msg):
        super().__init__(msg)

# ----------------------- Pickle-based RPC protocol ------------------------

import cloudpickle

class PickleResponse(Response):
    def serialize(self) -> bytes:
        return cloudpickle.dumps( if self.error else {
            'id': self.unique_id,
            'result': self.result,
        })

class PickleRequest(Request):
    def respond(self, result) -> PickleResponse:
        return PickleResponse(self.unique_id, result)

    def serialize(self) -> bytes:
        return cloudpickle.dumps({
            'id': self.unique_id,
            'method': self.method_name,
            'args': self.args,
            'kwargs': self.kwargs
        })

class PickleProtocol(Protocol):
    def create_request(self, method_name, args, kwargs):
        return PickleRequest(method_name, args, kwargs)

    def parse_request(self, bytes) -> Request:
        d = cloudpickle.loads(bytes)
        return PickleRequest(
            d['id'], d['method'],
            d['args'], d['kwargs']
        )

    def parse_response(self, bytes) -> Response:
        d = cloudpickle.loads(bytes)
        return PickleResponse(
            d['id'], d['result']
        )