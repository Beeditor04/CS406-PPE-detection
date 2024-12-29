from setuptools import setup, find_packages

setup(
    name='cs406_project',
    version='0.1',
    author="Beeditor",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'torchmetrics',
        'opencv-python',
        'Pillow',
    ],
    entry_points={
        'console_scripts': [
            'train_faster_rcnn=scripts.train_faster_rcnn:main',
            'inference=scripts.inference:main',
        ],
    },
)