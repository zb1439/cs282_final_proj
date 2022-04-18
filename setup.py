import os
import shutil
from setuptools import find_packages, setup

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(CUR_PATH, 'build')
if os.path.isdir(path):
    print('delete ', path)
    shutil.rmtree(path)
path = os.path.join(CUR_PATH, 'dist')
if os.path.isdir(path):
    print('delete ', path)
    shutil.rmtree(path)

head = "#!/bin/bash\n\n"
with open("rank_netease", "w") as f:
    f.write(head + f"python3 {os.path.join(CUR_PATH, 'main.py')} $@")

setup(
    name='netease_rank',
    author='Zhibo Fan',
    description="282 Final Project Netease Ranking System",
    version='0.1',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.20',
        'pandas',
        'matplotlib',
        'tqdm>4.29.0',
        'sklearn',
        'easydict',
        'colorama',
        'tabulate',
        "torch>=1.5.0",
        "urllib3"
    ],
    scripts=["rank_netease"],
)
