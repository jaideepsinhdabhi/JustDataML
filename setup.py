from setuptools import find_packages,setup
from typing import List


def get_requirements(file_path:str)->List[str]:
    """
    this function will give list of requriments
    """
    requirement = []
    with open(file_path, 'r') as file_obj:
        requirement=file_obj.readlines()
        requirement = [req.rstrip() for req in requirement]
        #print(requirement)
        if '-e .' in requirement:
            requirement.remove('-e .')
    return requirement


with open("README.md", "r") as f:
    long_description = f.read()

#print(get_requirements('requirements.txt'))

setup(
    name="JustDataML",
    version="0.0.1",
    author="Jaideepsinh Dabhi",
    author_email="jadieep.dabhi7603@gmail.com",
    description="This is a Machine Learning Tool Developed to Help Non Tech Students and Non Coders to try ML for there Data",
    package_dir={"":"JDML"},
    packages=find_packages(where="JDML"),
    #packages=find_packages(),
    #packages=find_packages(include=["JDML", "JDML.*"]),
    url="https://github.com/jaideepsinhdabhi/JustDataML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',  # Example license classifier
        'Programming Language :: Python :: 3.9',  # Indicate the supported Python versions
        'Operating System :: OS Independent'],
    install_requires=get_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'JDML=JDML:main'
        ]
    }
)


