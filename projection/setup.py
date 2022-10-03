from setuptools import setup, Extension

perspective = Extension(name="projection.perspective", sources=["perspective.c"])

setup (name = 'projection',
       version = '1.0',
       description = 'Projection modules',
       ext_modules = [perspective])
