from distutils.core import setup

setup(name='saliency-localizer',
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
      packages=['localizer']
)
