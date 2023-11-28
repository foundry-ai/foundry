from setuptools import setup

setup(
   name='tensorstore',
   version='0.1.50',
   description='Tensorstore mock',
   author='Daniel Pfrommer',
   author_email='dan.pfrommer@gmail.com',
   packages=['tensorstore'],  #same as name
   install_requires=[
       'wheel',
   ],
)
