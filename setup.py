from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        return requirements

setup(
    name='Diamond_Price_Pridiction',
    version='0.0.1',
    author= 'Yaseen Siddiqui',
    author_email= 'yaseensiddiqui36@gmail.com',
    install_requires= get_requirements('requirements.txt'),
    packages= find_packages()

)