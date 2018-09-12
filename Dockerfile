FROM jupyter/base-notebook
RUN pip install pew
RUN pip install -U pip
RUN pew new dsu

RUN pew workon dsu && pip install jupyter_contrib_nbextensions && \
	jupyter contrib nbextension install --user && jupyter nbextension enable codefolding/main \
	&& jupyter nbextension enable ExecutionTime/main

ADD requirements.txt /
RUN pip install -r /requirements.txt
