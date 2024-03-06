import sys
import builtins
from importlib import util, import_module
from typing import Any
from types import ModuleType

from prime.proxy import Proxy, Lineage, _client

PACKAGE = __package__
REAL_PACKAGE = PACKAGE.removeprefix('prime_')

ROOT_MODULE = import_module(REAL_PACKAGE)


""" ModuleWrapper serves as package/module """
class ModuleWrapper:
    def __init__(self, fullpath: str, module: ModuleType):
        self.fullpath = fullpath
        self.module = module
        self.is_package = hasattr(module, '__path__') # Hack
        
    def __getattr__(self, name: str):

        try:
            package_spec = util.find_spec(f'{self.fullpath}.{name}')
        except ModuleNotFoundError:
            package_spec = None
        
        if package_spec is not None:
            module = import_module(f'{self.fullpath}.{name}')
            return ModuleWrapper(f'{self.fullpath}.{name}', module)
        
        elif hasattr(self.module, name):
            obj = getattr(self.module, name)
            if isinstance(obj, ModuleType):
                return ModuleWrapper(f'{self.fullpath}.{name}', obj)
            elif name.startswith('__') and name.endswith('__'):
                return obj
            else:
                return ObjectWrapper(obj, False, f'{self.fullpath}.{name}')

        else:
            if self.is_package: 
                raise ModuleNotFoundError(f"No module named ")
            else: # module
                raise AttributeError(f"module '{self.fullpath}' has no attribute '{name}'")


""" ObjectWrapper serves as a class/instance/function """
class ObjectWrapper:
    __obj_wrapper__ = True

    @classmethod
    def is_proxy(cls, obj: Any) -> bool:
        if isinstance(obj, list):
            ret = any(isinstance(i, Proxy) for i in obj)
        elif isinstance(obj, Proxy):
            ret = True
        else:
            ret = False
            
        return ret

    # Only handle list and itself now
    @classmethod
    def unveil(cls, obj: Any) -> Any:
        if isinstance(obj, list):
            ret = []
            for i in obj:
                o = (i if (isinstance(i, Proxy) or not hasattr(i, '__obj_wrapper__')) 
                     else i.obj)
                ret.append(o)

        elif isinstance(obj, Proxy) or not hasattr(obj, '__obj_wrapper__'):
            ret = obj

        else:
            ret = obj
            
        return ret
                

    # is_dynamic indicates whether the wrapped object was dynamically instantiated
    def __init__(self, obj: object, is_dynamic: bool, path: str):
        self._proxy = None

        self.obj = obj
        self.is_dynamic = is_dynamic
        self.path = path

    def __call__(self, *args, **kwargs):

        proxy_in_args = any(ObjectWrapper.is_proxy(a) for a in args)
        proxy_in_kwargs = any(ObjectWrapper.is_proxy(v) for v in kwargs.values())

        new_args = [ObjectWrapper.unveil(a) for a in args]
        new_kwargs = {k:ObjectWrapper.unveil(v) for k, v in kwargs.items()}

        if proxy_in_args or proxy_in_kwargs:
            if self.is_dynamic:
                if self._proxy is None:
                    self._proxy = Proxy(_client.AllocateObj(self.obj))

                # Assume class is not dynamically defined
                # ref = _client.InvokeMethod(self._ref, '__call__', new_args, new_kwargs)

                lineage = Lineage(self._proxy, '__call__', new_args, new_kwargs)

            else:
                # ref = _client.InvokeMethod('', self.path, new_args, new_kwargs)
                lineage = Lineage('', self.path, new_args, new_kwargs)

            return Proxy(None, lineage)

        else:
            # This assumes functional
            obj = self.obj(*new_args, **new_kwargs)
            return ObjectWrapper(obj, True, f'{self.path}._obj')

    def __getattr__(self, name: str):
        obj = getattr(self.obj, name)

        if name.startswith('__') and name.endswith('__'):
            return obj
        else:
            return ObjectWrapper(obj, self.is_dynamic, f'{self.path}.{name}')

    def __getitem__(self, key):
        return self.obj[key]

    def __str__(self):
        return str(self.obj)


def __getattr__(name):
    # TODO: is this sound?
    package_spec = util.find_spec(f'{REAL_PACKAGE}.{name}')
    if package_spec is not None:
        module = import_module(f'{REAL_PACKAGE}.{name}')
        return ModuleWrapper(f'{REAL_PACKAGE}.{name}', module)
    
    elif hasattr(ROOT_MODULE, name):
        obj = getattr(ROOT_MODULE, name)
        if isinstance(obj, ModuleType):
            return ModuleWrapper(f'{REAL_PACKAGE}.{name}', obj)
        elif name.startswith('__') and name.endswith('__'):
            return obj
        else:
            return ObjectWrapper(obj, False, f'{REAL_PACKAGE}.{name}')

    else:
        raise ModuleNotFoundError(f"No module named '{REAL_PACKAGE}.{name}")

class ImportHook:
    def find_spec(self, fullname: str, path: str, target=None):
        if fullname.split('.')[0] == PACKAGE:
            return util.spec_from_loader(fullname, Loader(fullname))

class Loader:
    def __init__(self, fullname: str):
        self.fullname = fullname

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType):
        parts = self.fullname.split('.')
        if parts[0] != PACKAGE:
            raise Exception(f"Loader for '{PACKAGE}' invoked on {parts[0]}")

        current_module = builtins.__import__(parts[0])
        path = parts[0]
        for p in parts[1:]:
            path += '.' + p
            current_module = getattr(current_module, p)

            if not isinstance(current_module, ModuleWrapper):
                raise ModuleNotFoundError(f"No module named '{path}")
            
        sys.modules[self.fullname] = current_module

sys.meta_path.insert(0, ImportHook())