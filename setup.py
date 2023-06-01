from setuptools import setup

setup(
    name='projectp',
    version='0.1.0',
    packages=['projectp'],
    url='https://github.com/Shining-Future/project-p-inference',
    license='MIT',
    author='ValV',
    author_email='0x05a4@gmail.com',
    description='ProjectP inference package',
    install_requires=[
        'ensemble-boxes',
        'filetype',
        'matplotlib',
        'numpy',
        'onnxruntime',
        'opencv-python-headless',
        'pandas',
        'tqdm'
    ]
)
