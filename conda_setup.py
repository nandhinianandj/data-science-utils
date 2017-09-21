import settings

from distutils.core import setup, Extension
import distutils.command.bdist_conda

setup(
    name="datascienceutils",
    version=settings.version,
    distclass=distutils.command.bdist_conda.CondaDistribution,
    conda_buildnum=1,
    conda_features=['mkl'],
    conda_track_features=['mkl'],
    conda_buildstr=py36_1,
    conda_command_tests=False,
    conda_import_tests=False,
    conda_binary_reflection=False,
    conda_preserve_egg_dir=False,

)
