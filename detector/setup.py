from setuptools import setup, find_packages

setup(
    name='control_plane',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'redis>=5.0.0',
        'numpy>=1.22.2',
        'matplotlib>=3.7.0',
        'pandas>=1.5.0',
        'statsmodels>=0.13.0',
        'Rbeast>=0.1.19',
        'cvxpy>=1.5.3'
    ],
    entry_points = {
        'console_scripts': [
            'global_controller=control_plane.global_controller:main',
            'local_controller=control_plane.local_controller:start_local_controller',
        ],
    }
)
