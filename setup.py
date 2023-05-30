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
        'ensemble-boxes==1.0.9',
        'filetype==1.2.0',
        'matplotlib==3.6.2',
        'numpy==1.23.5',
        'onnxruntime==1.14.1',
        'opencv-python-headless==4.7.0.72',
        'pandas==1.5.0',
        'tqdm==4.62.3'
    ]
)
