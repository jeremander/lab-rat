# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['labrat', 'labrat.examples']

package_data = \
{'': ['*']}

install_requires = \
['colorlog', 'fancy_dataclass>=0.1', 'sqlalchemy', 'toml']

setup_kwargs = {
    'name': 'labrat',
    'version': '0.1.1',
    'description': 'A framework for running experiments in parallel.',
    'long_description': None,
    'author': 'Jeremy Silver',
    'author_email': 'jeremys@nessiness.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)

