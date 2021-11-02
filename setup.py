import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name="ngchihuan", # Replace with your username

    version="0.0.1",

    author="ChiHuan Nguyen",

    author_email="ngchihuan@gmail.com",

    description="Analyse sideband time evolution files and extract the Wigner Function",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/ngchihuan/WignerFunc_Measurement",

    packages=setuptools.find_packages(where="src"),

    package_dir={"": "src"},

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

    ],

    python_requires='>=3.5'


)
