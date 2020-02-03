from setuptools import setup

if __name__ == '__main__':
    setup(
        name='hawks',
        version='0.2.0',
        author='Cameron Shand',
        author_email='cameron.shand@manchester.ac.uk',
        packages=['hawks'],
        url='https://github.com/sea-shunned/hawks',
        license='MIT License',
        description='A package for generating synthetic clusters, with parameters to customize different aspects of the complexity of the cluster structure',
        long_description=open('README.rst').read(),
        long_description_content_type='text/x-rst',
        include_package_data=True,
        project_urls={
            "Documentation": "https://hawks.readthedocs.io/"
        },
        python_requires='>=3.6',
        install_requires=[
            "deap == 1.2.2",
            "matplotlib >= 3.0",
            "numpy >= 1.15",
            "orange3 >= 3.23",
            "pandas >= 0.23",
            "scikit-learn >= 0.20",
            "scipy >= 1.1",
            "seaborn >= 0.9.0",
            "tqdm >= 4.15",
        ],
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3.6",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ]
    )
