#
# Copyright (c) 2022
#

from __future__ import annotations

import os
import sys
import dill
import time
import shutil
from importlib.util import spec_from_file_location, module_from_spec
from typing import types, List, Dict, Any, Optional, Union, Callable

from prime.utils import logger
from prime.exceptions import *
from prime.hasref import FromRef, HasRef
from prime.emul import emulate
from prime.taint import *
from prime.static import check_violations
from prime.data import PrimeDataset

from prime.profiler import Profile

from pytorch_lightning import Trainer, LightningModule

import torch

import types
# sys.modules["test_language"] = types.ModuleType("test_language")

torch.utils.data.PrimeDataset = PrimeDataset

VAR = "VAR"

################################################################################
# Trusted Libraries                                                            #
################################################################################

TRUSTED_PKGS = {}


def _trust(pkg: str):
    module = __import__(pkg)

    global TRUSTED_PKGS
    TRUSTED_PKGS[pkg] = module


_trust("builtins")
_trust("torch")
_trust("pytorch_lightning")

################################################################################

################################################################################
# Custom Sample Initialization Function                                        #
################################################################################


def sample_init() -> ("torch.Tensor", "torch.Tensor"):
    samples = TRUSTED_PKGS["torch"].randint(0, 100, (10, 16, 16))
    labels = TRUSTED_PKGS["torch"].Tensor([0, 1] * 5)

    return (samples, labels)


################################################################################


def get_fullname(obj: type) -> str:
    try:
        return f"{obj.__module__}.{obj.__name__}"
    except:
        return ""


def path_exists(m: ModuleType, n: str) -> bool:
    if hasattr(m, "__path__"):
        path = "/".join(m.__path__[0].split("/")[:-1])
        if os.path.exists(f"{path}/{n}") or os.path.exists(f"{path}/{n}.py"):
            return True

    return False


def get_from(name: str) -> (str, FunctionType):
    pkg = name.split(".")[0]
    if pkg not in TRUSTED_PKGS:
        raise Exception(f"{name} is not from trusted packages")

    m = TRUSTED_PKGS[pkg]
    for n in name.split(".")[1:]:
        if hasattr(m, n):
            m = getattr(m, n)
        elif path_exists(m, n):
            m = __import__(f"{m.__name__}.{n}")
        else:
            raise Exception(f"{name} is not from trusted packages")

    return pkg, m


def from_trusted(tpe: type) -> bool:
    if tpe is type(None):
        return True

    try:
        name = f"{tpe.__module__}.{tpe.__name__}"
    except:
        return False

    pkg = name.split(".")[0]
    if pkg not in TRUSTED_PKGS:
        return False

    m = TRUSTED_PKGS[pkg]
    for n in name.split(".")[1:]:
        if hasattr(m, n):
            m = getattr(m, n)
        elif path_exists(m, n):
            m = __import__(f"{m.__name__}.{n}")
        else:
            return False

    if tpe is m:
        return True
    else:
        return True


class ExecutionRuntime:
    __initialized = False
    __ctx = {}
    __taints: TaintTracker = None

    def __init__(self, dn: Optional[str]):

        if self.__initialized:
            raise Exception("There can exist only one ExecutionRuntime")
        self.__init()

        self.ctr = 0

        self.dn = dn
        self.init_dataset()

    def init_dataset(self):
        if self.dn is None:
            samples, labels = sample_init()
        elif self.dn in ("mnist", "cifar10", "utkface", "chestxray"):
            _trust("torchvision")
            _trust("PIL")
            _trust("numpy")
            _trust("types")
            _trust("time")

            from ci_tests.mnist import mnist
            from eval_tests.datalib import cifar10, utkface, chestxray

            if self.dn == "mnist":
                samples, labels = mnist.sample_init()
            elif self.dn == "cifar10":
                samples, labels = cifar10.sample_init()
            elif self.dn == "utkface":
                samples, labels = utkface.sample_init()
            else:
                samples, labels = chestxray.sample_init()

            assert len(samples) == len(
                labels
            ), "Number of samples and labels mismatch"

            self.__taints.init(len(samples))

            s_tags = [UndefTag(0, i) for i in range(len(samples))]
            l_tags = [UndefTag(1, i) for i in range(len(labels))]

            s_tagsack = TagSack(s_tags)
            l_tagsack = TagSack(l_tags)

            self._add_to_ctx(samples, s_tagsack, "_SAMPLES")
            self._add_to_ctx(labels, l_tagsack, "_LABELS")

        elif self.dn in ("emotion", "wikipedia", "wikitext", "wmt16"):
            global lightning_transformers
            import lightning_transformers

            _trust("datasets")
            _trust("lightning_transformers")
            _trust("transformers")

            from eval_tests.datalib.nlp import (
                load_emotion,
                load_wikipedia,
                load_wikitext,
                load_wmt16,
            )

            if self.dn == "emotion":
                dataset = load_emotion()
            elif self.dn == "wikitext":
                dataset = load_wikitext()
            elif self.dn == "wikipedia":
                dataset = load_wikipedia()
            else:
                dataset = load_wmt16()

            self.__taints.init(len(dataset))

            tags = [UndefTag(0, i) for i in range(len(dataset))]
            tagsack = TagSack(tags)

            self._add_to_ctx(dataset, tagsack, "_DATASET")

        else:
            raise NotImplementedError(f"cannot test {self.ci}")

    @classmethod
    def __init(cls: ExecutionRuntime):
        cls.__initialized = True
        HasRef._set_ctx(cls.__ctx)
        cls.__taints = TaintTracker()

    # TODO: Need lock?
    def _add_to_ctx(
        self, obj: Any, tag: Union[Tag, TagSack], name: str = None
    ) -> str:
        if not name:
            name = f"{VAR}{self.ctr}"
            self.ctr += 1

        self.__ctx[name] = obj
        self.__taints[name] = tag

        logger.debug(f"{name:<8}: {hex(id(obj))} <--- {tag}")
        return name

    # TODO: Need lock?
    def _del_from_ctx(self, name: str):

        logger.debug(f"{name}")
        del self.__ctx[name]
        del self.__taints[name]

    def _from_ctx(self, tpe: type) -> bool:
        if get_fullname(tpe) not in self.__ctx:
            return False

        if tpe is not self.__ctx[get_fullname(tpe)]:
            return False

        return True

    # TODO: Still need to check a class instance from trusted package does
    # not contain malicious method
    def _deserialize(
        self, val: bytes, useref: bool = True
    ) -> (Union[Tag, TagSack], Any):

        # TODO: is it thread-safe?

        HasRef._set_useref(useref)

        fromref = FromRef()
        HasRef._set_fromref(fromref)
        obj = dill.loads(val)

        tpe = type(obj)

        obj_in_ctx = len(fromref) == 1 and self.__ctx[fromref[0]] is obj

        # assert (
        #     not isinstance(obj, Callable) or isinstance(obj, LightningModule) 
        #     or obj_in_ctx
        # ), f"invalid type: {obj}"
        assert (
            from_trusted(tpe) or self._from_ctx(tpe) or obj_in_ctx
        ), f"type not trusted: {obj}({tpe})"

        if obj_in_ctx:
            ref = fromref[0]
            tag = self.__taints[ref]

        elif fromref:
            if tpe is list:
                tags = [self.__taints[r] for r in fromref]
                tag = TagSack(tags)

                for r in fromref:
                    self.__taints[r] = DangerTag()

            else:
                # TODO
                raise PrimeNotAllowedError(
                    f"exporting variable containing references is not allowed"
                )

            # for ref in fromref:
            #     self.__taints[ref] = DangerTag()

            # tag = DangerTag() # TODO: Can give weaker tag?

        else:
            tag = SafeTag(hash(val))

        return tag, obj

    @catch_xcpt(False)
    def ExportDef(
        self, fullname: str, tpe: bytes, source: str, profile=False
    ) -> str:
        tpe = dill.loads(tpe)

        assert tpe in (type, types.FunctionType), f"not supported type: {tpe}"

        logger.debug(f"{fullname} ({tpe})")

        # TODO: Sandbox codes
        # Malicious codes can change states of global variables

        if fullname.startswith("__main__"):
            # NOTE: Do not support nested definition
            assert len(fullname.split(".")) == 2
            name = fullname.split(".")[1]
            assert name not in globals()

            try:
                venv = {"__name__": "__main__"}
                venv["__name__"] = "__main__"

                for pkg, module in TRUSTED_PKGS.items():
                    venv[pkg] = module

                exec(source, venv)

            except Exception as e:
                raise UserError(e)

            obj = venv[name]
            globals()[name] = obj

        else:
            raise NotImplementedError()
            # module_path = fullname.split('.')[:-1]
            # name = fullname.split('.')[-1]

            # for i in range(len(module_path) -1):
            #     p = f'/tmp/{"/".join(module_path[:i+1])}'
            #     os.mkdir(f'{p}')

            #     with open(p + '/__init__.py', 'w') as fd:
            #         fd.write(f'import {".".join(module_path[:i+2])}')

            # with open(f'/tmp/{"/".join(module_path)}.py', 'w') as fd:
            #     fd.write(source)

            # root = module_path[0]
            # path = (f'/tmp/{root}.py' if len(module_path) == 1
            #         else f'/tmp/{root}/__init__.py')
            # spec = spec_from_file_location(root, path)
            # module = module_from_spec(spec)

            # try:
            #     for pkg, m in TRUSTED_PKGS.items():
            #         setattr(module, pkg, m)

            #     spec.loader.exec_module(module)
            # except Exception as e:
            #     raise UserError(e)

            # sys.modules[root] = module

            # if len(module_path) == 1: os.remove(f'/tmp/{root}.py')
            # else: shutil.rmtree(f'/tmp/{root}')

            # obj = module
            # for n in fullname.split('.')[1:]:
            #     obj = getattr(obj, n)

        self._add_to_ctx(obj, SafeTag(hash(source)), fullname)
        return name, None

    def DeleteObj(self, name: str):
        self._del_from_ctx(name)

    @catch_xcpt(False)
    def AllocateObj(self, val: bytes, profile=False) -> str:

        # AllocateObj does not allow using reference
        tag, obj = self._deserialize(val, False)

        # TODO: AllocateObj is only for invoking a instance function
        # assert callable(obj), f'cannot allocate non-callable: {obj}'

        logger.debug(f"{hex(id(obj))}={str(obj)[0:10]}...")
        name = self._add_to_ctx(obj, tag)
        return name, None

    @catch_xcpt(False)
    def InvokeMethod(
        self,
        obj: str,
        method: str,
        args: List[bytes],
        kwargs: Dict[str, bytes],
        ref: Optional[str] = None,
        profile=False,
    ) -> Union[str, bytes]:

        pf = Profile()
        now = time.time()

        # FIXME: This is only for test purpose ##############################
        # Remove this before release! #######################################
        if obj == "" and method == "get_tag":
            tag, _ = self._deserialize(args[0])
            with open("/tmp/x.txt", "w") as fd:
                fd.write(str(tag))

            return dill.dumps(None), None

        ####################################################################

        if obj:  # obj is allocated in context
            self_tag = self.__taints[obj]
            obj = self.__ctx[obj]  # Class instance or instance method

            try:
                method = getattr(obj, method)
            except Exception as e:
                raise UserError(e)

            if callable(obj) and hasattr(obj, "__self__"):  # instance method
                # TODO: This assumes obj.__self__ is class instance
                _self = obj.__self__
                module = (
                    _self.__module__ if hasattr(_self, "__module__") else None
                )
                module = module.split('.')[0] if module else None

            elif isinstance(self_tag, TagSackIterator):
                # TagSackIterator can only be constructed from Tensor
                module = "torch"

            else:
                module = obj.__module__ if hasattr(obj, "__module__") else None
                module = module.split('.')[0] if module else None

        else:  # obj is not specified
            self_tag = None

            if method in self.__ctx:
                method = self.__ctx[method]
                module = "__main__"  # Exported functions only

            else:
                module, method = get_from(method)

        t_args = [self._deserialize(i) for i in args]
        t_kwargs = {k: self._deserialize(v) for k, v in kwargs.items()}

        args = [i[1] for i in t_args]
        kwargs = {k: v[1] for k, v in t_kwargs.items()}

        tags = [i[0] for i in t_args]
        kwtags = {k: v[0] for k, v in t_kwargs.items()}

        cur = time.time()
        pf.serialize += cur - now
        now = cur

        logger.debug(f"{method}")
        try:
            out = emulate(method, obj)(*args, **kwargs)
        except Exception as e:
            raise UserError(e)

        cur = time.time()
        pf.op += cur - now
        now = cur

        try:
            # FIXME: This is only for test purpose ##############################
            # Remove this before release! #######################################
            if hasattr(method, "__name__") and method.__name__ == "output":
                tag = DangerTag()

            elif method.__name__ == "getattr" and args[1] == "__name__":
                tag = DangerTag()
            ####################################################################

            else:
                tag = taint(
                    method, module, args, kwargs, self_tag, tags, kwtags
                )
        except TagError as e:
            raise PrimeNotAllowedError(e.msg)

        if out is NotImplemented:
            raise NotImplementedOutputError()

        cur = time.time()
        pf.taint += cur - now
        now = cur

        logger.debug(f"{hex(id(out))}={str(out)[0:10]}...")
        ret = (
            dill.dumps(out)
            if isinstance(tag, Tag)
            and tag.is_safe()
            and not isinstance(out, Trainer)
            else self._add_to_ctx(out, tag, ref)
        )

        cur = time.time()
        pf.serialize += cur - now
        now = cur

        return ret, pf

    @catch_xcpt(False)
    def ExportModel(self, fullname: str, source: str, profile=False) -> str:
        # TODO: Sandbox codes
        # Malicious codes can change states of global variables
        violations = check_violations(source)

        if violations:
            msg = "Violation detected in model definition\n" + "\n".join(
                f"[{i}] {v}" for i, v in enumerate(violations)
            )

            raise PrimeNotAllowedError(msg)

        if fullname.startswith("__main__"):
            # NOTE: Do not support nested definition
            assert len(fullname.split(".")) == 2
            name = fullname.split(".")[1]
            assert name not in globals()

            try:
                venv = {"__name__": "__main__"}

                for pkg, module in TRUSTED_PKGS.items():
                    venv[pkg] = module

                exec(source, venv)

            except Exception as e:
                raise UserError(e)

            obj = venv[name]
            globals()[name] = obj

        else:
            # e.g., ci_tests.cifar_10.CIFARModule
            module_path = fullname.split(".")[:-1]
            name = fullname.split(".")[-1]

            # Build module directory structure, e.g., /tmp/ci_tests/cifar_10.py
            for i in range(len(module_path) - 1):
                m_path, m = module_path[: i + 1], module_path[i + 1]

                p = f'/tmp/{"/".join(m_path)}'
                os.mkdir(f"{p}")

                with open(p + "/__init__.py", "w") as fd:
                    fd.write(f'from {".".join(m_path)} import {m}')

            # Write final module which contains real source code + imports
            with open(f'/tmp/{"/".join(module_path)}.py', "w") as fd:
                fd.writelines(
                    [f"import {pkg}\n" for pkg in TRUSTED_PKGS.keys()] + ["\n"]
                )
                fd.write(source)

            # Import root module
            root = module_path[0]
            path = (
                f"/tmp/{root}.py"
                if len(module_path) == 1
                else f"/tmp/{root}/__init__.py"
            )
            spec = spec_from_file_location(root, path)
            module = module_from_spec(spec)

            try:
                spec.loader.exec_module(module)
            except Exception as e:
                raise UserError(e)

            sys.modules[root] = module

            # Remove tmp files
            if len(module_path) == 1:
                os.remove(f"/tmp/{root}.py")
            else:
                shutil.rmtree(f"/tmp/{root}")

            obj = module
            for n in fullname.split(".")[1:]:
                obj = getattr(obj, n)

        self._add_to_ctx(obj, SafeTag(hash(source)), fullname)
        return name, None

    @catch_xcpt(True)
    def FitModel(
        self,
        trainer: str,
        model: str,
        args: List[bytes],
        kwargs: Dict[str, bytes],
        profile=False,
    ) -> Union[str, bytes]:

        import pytorch_lightning as pl

        trainer_tag = self.__taints[trainer]
        trainer: pl.Trainer = self.__ctx[trainer]

        model_tag = self.__taints[model]
        model: pl.LightningModule = self.__ctx[model]

        if not trainer_tag.is_safe() or not model_tag.is_safe():
            raise PrimeNotAllowedError(
                "cannot fit model with unsafe trainer & model"
            )

        try:
            dataloader = kwargs.pop("train_dataloaders")
        except KeyError:
            dataloader = args.pop(0)
        except:
            raise PrimeNotAllowedError("train_dataloader is not provided")

        tag, dataloader = self._deserialize(dataloader, True)

        if not tag.is_frozen():
            raise PrimeNotAllowedError(
                "cannot fit model with un-sanitized dataloader"
            )

        args = [self._deserialize(i, False)[1] for i in args]
        kwargs = {k: self._deserialize(v, False)[1] for k, v in kwargs.items()}

        torch.set_num_threads(1)

        try:
            trainer.fit(model, dataloader, *args, **kwargs)
        except Exception as e:
            raise UserError(e)

        from io import BytesIO

        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        model = buffer.getvalue()
        buffer.close()

        # model = dill.dumps(model)
        return model, None
