'''
Cython automatic extension builder/manager

Inspired by IPython's Cython cell magics, see:
https://github.com/ipython/ipython/blob/master/IPython/extensions/cythonmagic.py
'''
import imp
import os
import sys
import time
try:
    import msvcrt
except ImportError:
    msvcrt = None
    import fcntl

try:
    import hashlib
except ImportError:
    import md5 as hashlib

from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext

try:
    import Cython
    import Cython.Compiler as Cython_Compiler
    import Cython.Build as Cython_Build
except ImportError:
    Cython = None

from brian_genn_version.utils.logger import std_silent
from brian_genn_version.utils.stringtools import deindent
from brian_genn_version.core.preferences import prefs

__all__ = ['cython_extension_manager']


class CythonExtensionManager(object):
    def __init__(self):
        self._code_cache = {}
        
    def create_extension(self, code, force=False, name=None,
                         include_dirs=None,
                         library_dirs=None,
                         runtime_library_dirs=None,
                         extra_compile_args=None,
                         extra_link_args=None,
                         libraries=None,
                         compiler=None,
                         ):

        if Cython is None:
            raise ImportError('Cython is not available')

        code = deindent(code)

        lib_dir = os.path.expanduser('~/.brian/cython_extensions')
        try:
            os.makedirs(lib_dir)
        except OSError:
            if not os.path.exists(lib_dir):
                raise

        key = code, sys.version_info, sys.executable, Cython.__version__
            
        if force:
            # Force a new module name by adding the current time to the
            # key which is hashed to determine the module name.
            key += time.time(),            

        if key in self._code_cache:
            return self._code_cache[key]

        if name is not None:
            module_name = name#py3compat.unicode_to_str(args.name)
        else:
            module_name = "_cython_magic_" + hashlib.md5(str(key).encode('utf-8')).hexdigest()



        module_path = os.path.join(lib_dir, module_name + self.so_ext)

        if prefs['codegen.runtime.cython.multiprocess_safe']:
            lock_file = os.path.join(lib_dir, module_name + '.lock')
            with open(lock_file, 'w') as f:
                if msvcrt:
                    msvcrt.locking(f.fileno(), msvcrt.LK_RLCK,
                                   os.stat(lock_file).st_size)
                else:
                    fcntl.flock(f, fcntl.LOCK_EX)
                return self._load_module(module_path, include_dirs,
                                         library_dirs,
                                         extra_compile_args, extra_link_args,
                                         libraries, code, lib_dir, module_name,
                                         runtime_library_dirs, compiler, key)
        else:
            return self._load_module(module_path, include_dirs, library_dirs,
                                     extra_compile_args, extra_link_args,
                                     libraries, code, lib_dir, module_name,
                                     runtime_library_dirs, compiler, key)

    @property
    def so_ext(self):
        """The extension suffix for compiled modules."""
        try:
            return self._so_ext
        except AttributeError:
            self._so_ext = self._get_build_extension().get_ext_filename('')
            return self._so_ext

    def _clear_distutils_mkpath_cache(self):
        """clear distutils mkpath cache
        
        prevents distutils from skipping re-creation of dirs that have been removed
        """
        try:
            from distutils.dir_util import _path_created
        except ImportError:
            pass
        else:
            _path_created.clear()


    def _get_build_extension(self, compiler=None):
        self._clear_distutils_mkpath_cache()
        dist = Distribution()
        config_files = dist.find_config_files()
        try:
            config_files.remove('setup.cfg')
        except ValueError:
            pass
        dist.parse_config_files(config_files)
        build_extension = build_ext(dist)
        if compiler is not None:
            build_extension.compiler = compiler
        build_extension.finalize_options()
        return build_extension

    
    def _load_module(self, module_path, include_dirs, library_dirs,
                     extra_compile_args, extra_link_args, libraries, code,
                     lib_dir, module_name, runtime_library_dirs, compiler,
                     key):
        have_module = os.path.isfile(module_path)

        if not have_module:
            if include_dirs is None:
                include_dirs = []
            if library_dirs is None:
                library_dirs = []
            if extra_compile_args is None:
                extra_compile_args = []
            if extra_link_args is None:
                extra_link_args = []
            if libraries is None:
                libraries = []

            c_include_dirs = include_dirs
            if 'numpy' in code:
                import numpy
                c_include_dirs.append(numpy.get_include())

            # TODO: We should probably have a special folder just for header
            # files that are shared between different codegen targets
            import brian_genn_version.synapses as synapses
            synapses_dir = os.path.dirname(synapses.__file__)
            c_include_dirs.append(synapses_dir)

            pyx_file = os.path.join(lib_dir, module_name + '.pyx')
            # ignore Python 3 unicode stuff for the moment
            #pyx_file = py3compat.cast_bytes_py2(pyx_file, encoding=sys.getfilesystemencoding())
            #with io.open(pyx_file, 'w') as f:#, encoding='utf-8') as f:
            #    f.write(code)
            open(pyx_file, 'w').write(code)

            extension = Extension(
                name=module_name,
                sources=[pyx_file],
                include_dirs=c_include_dirs,
                library_dirs=library_dirs,
                runtime_library_dirs=runtime_library_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                libraries=libraries,
                language='c++',
                )
            build_extension = self._get_build_extension(compiler=compiler)
            try:
                opts = dict(
                    quiet=True,
                    annotate=False,
                    force=True,
                    )
                # suppresses the output on stdout
                with std_silent():
                    build_extension.extensions = Cython_Build.cythonize([extension], **opts)

                    build_extension.build_temp = os.path.dirname(pyx_file)
                    build_extension.build_lib = lib_dir
                    build_extension.run()
            except Cython_Compiler.Errors.CompileError:
                return

        module = imp.load_dynamic(module_name, module_path)
        self._code_cache[key] = module
        return module
        #self._import_all(module)
        
cython_extension_manager = CythonExtensionManager()


if __name__=='__main__':
    code = '''
    def f(double x):
        return x*x
    '''
    man = CythonExtensionManager()
    mod = man.create_extension(code)
    print mod.f(2)
    