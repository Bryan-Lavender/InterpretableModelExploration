from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os

extensions = [
    Extension(
        name="DecisionTree",
        sources=["DecisionTree.py"],
    ),
    Extension(
        name="Backend_src.Criterion",
        sources=["Backend_src/Criterion.py"],
    ),
    Extension(
        name="Backend_src.DecisionTreeCreator",
        sources=["Backend_src/DecisionTreeCreator.py"],
    ),
    Extension(
        name="Backend_src.SplittingFunctions",
        sources=["Backend_src/SplittingFunctions.py"],
    ),
    Extension(
        name="Backend_src.ShouldCreateLeaf",
        sources=["Backend_src/ShouldCreateLeaf.py"],
    ),
    Extension(
        name="Backend_src.WeighingFunctions",
        sources=["Backend_src/WeighingFunctions.py"],
    )
]

setup(
    name="CompiledTree",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)