"""
Setup script for TensorFuse Python bindings.
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import pybind11

class CMakeBuild(build_ext):
    """Custom build extension that uses CMake."""
    
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build TensorFuse")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DTENSORFUSE_BUILD_PYTHON=ON',
            '-DTENSORFUSE_BUILD_TESTS=OFF',
            '-DTENSORFUSE_BUILD_BENCHMARKS=OFF',
        ]
        
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        build_args += ['--', '-j4']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
            env=env
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )

# Define the extension
ext_modules = [
    Extension(
        'tensorfuse',
        [],
        include_dirs=[
            pybind11.get_cmake_dir(),
        ],
        language='c++',
        sourcedir='.',
    ),
]

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tensorfuse",
    version="1.0.0",
    author="TensorFuse Contributors",
    author_email="contact@tensorfuse.ai",
    description="Tensor-Core-Optimized Transformer Inference Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorfuse/tensorfuse",
    packages=["tensorfuse"],
    package_dir={"": "src/python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pybind11>=2.10.0",
    ],
    extras_require={
        "torch": ["torch>=1.12.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
        ],
    },
    zip_safe=False,
) 