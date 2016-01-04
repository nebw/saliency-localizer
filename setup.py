from distutils.core import setup

setup(name='saliceny-localizer',
      version='0.0.1',
      description='',
      install_requires=[
            "numpy>=1.9",
            "keras>=0.3.0",
            "scikit-image>=0.11.3",
            "theano",
            "seaborn",
            "h5py",
      ],
      packages=['localizer']
)
