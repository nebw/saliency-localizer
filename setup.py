from distutils.core import setup

setup(
    name='saliency-localizer',
    version='0.0.1',
    description='',
    install_requires=[
          "numpy>=1.9",
          "keras>=1.0.1",
          "scikit-image>=0.11.3",
          "theano",
          "seaborn",
          "h5py",
    ],
    entry_points={
        'console_scripts': [
            'bb_find_tags=localizer.scripts.find_tags:main',
            'bb_build_tag_dataset=localizer.scripts.build_tag_dataset:main',
        ],
    },
    packages=['localizer']
)
