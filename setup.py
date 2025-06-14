from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='Karasin_JKU_task1',
    version='0.1.0',
    description='Inference package for DCASE25 Task 1 (based on https://github.com/CPJKU/dcase2025_task1_inference)',
    author='MALACH25 Task1 team',
    author_email="k12213736@students.jku.at",
    packages=find_packages(),  # This auto-discovers the inner folder
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'Karasin_JKU_task1': ["resources/*.wav", 'ckpts/*.ckpt'],
    },
    python_requires='>=3.13',
)
