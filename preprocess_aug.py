import multiprocessing as mp
import os
import sys

CONTINUE_PROCESS=True
PYTHON="python3"
EXTRACTOR="JavaExtractor/extract.py"
NUM_THREADS=4
REMOVE="rm"
# REMOVE="del"

def preprocess_dir(path:str):
    # analysize both
    if (not CONTINUE_PROCESS) or (not os.path.exists(os.path.join(path,"both.test.c2v"))):
        os.system(PYTHON + " " + EXTRACTOR + " --dir " + path + " --max_path_length 8 --max_path_width 2 --num_threads " + str(NUM_THREADS) +
                  " --jar JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar > " + path + "/augtempboth")
        os.system(PYTHON + " preprocess_test_batch.py --test_data " +
                  path + "/augtempboth --max_contexts 200 --dict_file data/java14m/java14m --output_name " + path + "/both")
        os.remove(path + "/augtempboth")

    # analysize original
    if (not CONTINUE_PROCESS) or (not os.path.exists(os.path.join(path, "original.test.c2v"))):
        path = path
        for f in os.listdir(path + "/src"):
            if f[-13: -5] != "_mutants":
                f = os.path.join(path,"src",f)
                os.system(PYTHON + " " + EXTRACTOR + " --file " + f + " --max_path_length 8 --max_path_width 2 --num_threads " + str(NUM_THREADS) +
                    " --jar JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar > " + path + "/augtemporigin")
                os.system(PYTHON + " preprocess_test_batch.py --test_data " +
                          path + "/augtemporigin --max_contexts 200 --dict_file data/java14m/java14m --output_name " + path + "/original")
                os.remove(path + "/augtemporigin")


if __name__ == "__main__":

    mypath = sys.argv[1]
    alldirs = [os.path.join(mypath, f).replace("\\", "/") for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, f))]

    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(preprocess_dir, alldirs)