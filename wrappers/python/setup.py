from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class CMakeExtension(Extension):

    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(_build_ext):

    def run(self):
        cmake_extensions = []
        rest_extensions = []
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                cmake_extensions.append(ext)
            else:
                rest_extensions.append(ext)
        self.extensions = rest_extensions
        super().run()
        for ext in cmake_extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        package_dir = Path(__file__).parent
        root_dir = (package_dir / '../..').resolve()

        build_dir = Path(self.build_temp)
        install_dir = Path(self.build_lib) / 'nntile'
        if self.inplace:
            install_dir = package_dir / 'nntile'
        install_dir = install_dir.resolve()

        # Reconfigure cmake with directory to "install" extension library.
        cmd = ('cmake', '-S', str(root_dir), '-B', str(build_dir),
               f'-DNNTILE_CORE_SYMLINK_DIR={install_dir}')
        self.spawn(cmd)

        # Make a symlink to `nntile_core` shared library.
        cmd = ('cmake', '--build', str(build_dir), '--target',
               'nntile-symlink')
        self.spawn(cmd)

    def spawn(self, cmd, search_path=1, level=1):
        if self.verbose:
            command = ' '.join(cmd)
            print(f'build_ext: run command: ({command})')
        super().spawn(cmd, search_path, level)


setup(ext_modules=[CMakeExtension('nntile.nntile_core')],
      cmdclass={'build_ext': build_ext})
