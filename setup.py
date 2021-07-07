import os

import setuptools
import versioneer


here = os.path.abspath(os.path.dirname(__file__))


def get_requirements(require_name=None):
    prefix = require_name + '_' if require_name is not None else ''
    with open(os.path.join(here, prefix + 'requirements.txt'), encoding='utf-8') as f:
        return f.read().strip().split('\n')


setuptools.setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=get_requirements(),
    extras_require={
        'dev': get_requirements('dev'),
        'docs': get_requirements('docs'),
    }
)
