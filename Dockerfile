FROM jupyter/base-notebook
RUN pip install pew
RUN pew new dsu
RUN pew workon dsu && pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable ExecutionTime/main
