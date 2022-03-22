from textwrap import dedent

from nuitka.plugins.PluginBase import NuitkaPluginBase


class NuitkaPluginFixBuild(NuitkaPluginBase):
    plugin_name = "fix-build"

    @staticmethod
    def onModuleSourceCode(module_name, source_code):
        if module_name == "torch.utils.data._typing":
            source_code = source_code.replace(
                "'__init_subclass__': _dp_init_subclass",
                "'__init_subclass__': classmethod(_dp_init_subclass)",
            )
        elif module_name == "numba.core.decorators":
            source_code = dedent(
                """\
                from numba.stencils.stencil import stencil
                def _wrapper(f):
                    return f
                def jit(*args, **kwargs):
                    return _wrapper
                def generated_jit(*args, **kwargs):
                    return _wrapper
                def njit(*args, **kwargs):
                    return _wrapper
                def cfunc(*args, **kwargs):
                    return _wrapper
                def jit_module(*args, **kwargs):
                    pass
                """
            )
        elif module_name == "torch._jit_internal":
            source_code = source_code.replace(
                'warnings.warn(f"Unable to retrieve source',
                "#",
            )
        elif module_name == "librosa.decompose":
            source_code = source_code.replace("import sklearn.decomposition", "#")
        elif module_name == "librosa.segment":
            source_code = source_code.replace("import sklearn", "#")
        return source_code