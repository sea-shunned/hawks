from setuptools import setup

if __name__ == '__main__':
    setup(
        name='hawks',
        version='0.0.1',
        author='Cameron Shand',
        author_email='cameron.shand@manchester.ac.uk',
        packages=['hawks'],
        url='https://github.com/sea-shunned/hawks',
        license='MIT License',
        description='A package for generating synthetic clusters, with parameters to customize different aspects of the complexity of the cluster structure',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        include_package_data=True,
        python_requires='>=3.6',
        install_requires=[
            "deap == 1.2.2",
            "matplotlib >= 2.1",
            "numpy >= 1.15",
            "pandas >= 0.23",
            "scikit-learn >= 0.20",
            "scipy >= 1.1",
            "tqdm >= 4.15"
        ],
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3.6",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ]
    )
