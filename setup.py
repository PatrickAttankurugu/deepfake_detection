from setuptools import setup, find_packages

setup(
    name="deepfake_detector",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'django>=3.2.0',
        'torch>=1.7.0',
        'timm>=0.4.12',
        'numpy>=1.19.2',
        'opencv-python>=4.4.0',
        'Pillow>=8.0.0',
        'albumentations>=0.5.2',
        'facenet-pytorch>=2.5.2',
        'gdown>=4.4.0',
    ],
    author="Agregar Technologies",
    description="A DeepFake detection package for Django applications",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PatrickAttankurugu/deepfake_detection",
    python_requires='>=3.8',
    classifiers=[
        'Framework :: Django',
        'Programming Language :: Python :: 3',
        'Topic :: Security',
    ],
    test_suite='deepfake_detector.tests',
)
