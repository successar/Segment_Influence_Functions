import os 

if "NER_INF_BASEDIR" in os.environ:
    base_dir = os.environ["NER_INF_BASEDIR"]
else :
    base_dir = "."

add_base_dir = lambda x : os.path.join(base_dir, os.path.normpath(x))