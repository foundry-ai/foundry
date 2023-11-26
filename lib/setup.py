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
       'flax',
       'ipython>=8.18.0',
       'ipykernel',
       'ffmpegio>=0.8.5'
   ],
   scripts=['scripts/launch']
)
