# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup


VERSION = "0.11.2.dev0"

extras = {}
extras["quality"] = [
    "black",  # doc-builder has an implicit dependency on Black, see huggingface/doc-builder#434
    "hf-doc-builder",
    "ruff~=0.4.8",
]
extras["docs_specific"] = [
    "black",  # doc-builder has an implicit dependency on Black, see huggingface/doc-builder#434
    "hf-doc-builder",
]
extras["dev"] = extras["quality"] + extras["docs_specific"]
extras["test"] = extras["dev"] + [
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "parameterized",
    "datasets",
    "diffusers<0.21.0",
    "scipy",
]

setup(
    name="peft",
    version=VERSION,
    description="Parameter-Efficient Fine-Tuning (PEFT)",
    license_files=["LICENSE"],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="The HuggingFace team",
    author_email="sourab@huggingface.co",
    url="https://github.com/huggingface/peft",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"peft": ["py.typed", "tuners/boft/fbd/fbd_cuda.cpp", "tuners/boft/fbd/fbd_cuda_kernel.cu"]},
    entry_points={},
    python_requires=">=3.8.0",
    install_requires=[
        "numpy>=1.17",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=1.13.0",
        "transformers",
        "tqdm",
        "accelerate>=0.21.0",
        "safetensors",
        "huggingface_hub>=0.17.0",
    ],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
