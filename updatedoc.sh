python3 setup.py install
cd docs
make clean 
make html
cd ../
open docs/_build/html/index.html 