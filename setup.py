from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='Experiments on label shift domain adaptation',
          url='NA',
          version='0.1.0.0',
          packages=['labelshiftexperiments'],
          setup_requires=[],
          install_requires=['numpy>=1.9',
                            'scikit-learn>=0.20.0',
                            'scipy>=1.1.0'],
          scripts=[],
          name='labelshiftexperiments')
