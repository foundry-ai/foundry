from setuptools import setup

setup(
   name='stanza',
   version='0.0.1',
   description='An ML toolkit',
   author='Daniel Pfrommer',
   author_email='dan.pfrommer@gmail.com',
   packages=['stanza'],  #same as name
   install_requires=[
       'wheel',
       'jax',
       'flax==0.7.5',
       'ipython>=8.18.0',
       'matplotlib>=3.8.2',
       'ipykernel',
       'ffmpegio>=0.8.5'
   ],
   scripts=['scripts/launch']
)
