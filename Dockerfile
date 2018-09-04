FROM jupyter/base-notebook
RUN pip install pew
RUN pew new dsu
RUN pew workon dsu && pip install jupyter_contrib_nbextensions
RUN pip install -U pip
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable ExecutionTime/main

ADD requirements.txt /
RUN pip install -r /requirements.txt
