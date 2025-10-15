import os
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
import subprocess

from git import Repo as GitRepo, GitCommandError
from dvc.repo import Repo as DVCRepo

from src.Classification.utils.common import read_yaml, write_yaml, get_local_version
from src.Classification.constants import *
from src.Classification import logger

app = FastAPI()
git_repo = GitRepo(".")
dvc_repo = DVCRepo(".")

BRANCH_NAME = "tgv"
COUNT_DATASET_PATH = os.path.join("artifacts", "data_ingestion", "CountDatasets.png")
TRAINING_LOG_PATH = os.path.join("artifacts", "training", "training.log")
TRAINING_GRAPH_PATH = os.path.join("artifacts", "training", "TrainingGraph.png")
CONFUSION_MATRIX_PATH = os.path.join("artifacts", "confusion_matrix.png")
DATASET_PATH = os.path.join("artifacts", "data_ingestion", "dataset.zip")
ONNX_MODEL_PATH = os.path.join("artifacts", "training", "model.onnx")  

# Helper: run cmd
def run_command(command: list):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    if result.returncode != 0:
        message = f"Error: {result.stderr}"
        logger.error(message)
        raise HTTPException(status_code=500, detail=message)
    return result.stdout

########################################## DVC ##########################################
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Training API!"}

@app.post("/Reproduce-pipeline")
def dvc_reproduce_pipeline():
    try:
        dvc_repo.reproduce()
        message = "Pipeline has been successfully reproduced!"
        logger.info(message)
        return {"message": message}
        # return True
    except Exception as e:
        logger.error(f"Error while reproducing the pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")  
    finally:
        dvc_repo.close()

class ReproStageRequest(BaseModel):
    stage_name: str

@app.post("/Reproduce-stage")
def dvc_reproduce_stage(request: ReproStageRequest):
    try:
        dvc_repo.reproduce(request.stage_name)
        message = f"Stage '{request.stage_name}' has been successfully reproduced!"
        logger.info(message)
        return {"message": message}
    except Exception as e:
        logger.error(f"Error while reproducing stage '{request.stage_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    finally:
        dvc_repo.close()


def dvc_push_data():
    """
    Pushes data to the remote storage using DVC.

    Returns:
        bool: True if the data was successfully pushed, False otherwise.
    """
    try:
        dvc_repo.push()
        message = "Data successfully pushed to remote storage!"
        logger.info(message)
        return True
    except Exception as e:
        logger.error(f"Error while pushing data: {e}")
        return False
    finally:
        dvc_repo.close()


def dvc_pull_data():
    try:
        dvc_repo.pull(force=True)
        message = "Data successfully pulled from remote storage!"
        logger.info(message)
        return True
    except Exception as e:
        logger.error(f"Error during pull: {e}")
        return False
    finally:
        dvc_repo.close()


# Get status DVC
@app.get("/Get-status")
def dvc_get_status():
    try:
        status = dvc_repo.status()
        logger.info(status)
        return status
    except Exception as e:
        logger.error(f"Error while checking status: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    finally:
        dvc_repo.close()


# Get metrics DVC
@app.get("/Get-metrics")
def dvc_get_metrics():
    try:
        metrics = dvc_repo.metrics.show()
        logger.info(metrics)
        return metrics
    except Exception as e:
        logger.error(f"Error while retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    finally:
        dvc_repo.close()


class RefDiffRequest(BaseModel):
    old_ref: str
    new_ref: str

@app.post("/Diff-metrics")
def dvc_diff_metrics(request: RefDiffRequest):
    try:
        diff = dvc_repo.metrics.diff(request.old_ref, request.new_ref)
        logger.info(diff)
        return diff
    except Exception as e:
        logger.error(f"Error while checking metrics diff: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    finally:
        dvc_repo.close()

# Get stages in DVC
@app.get("/Get-stages")
def dvc_get_stages():
    command = "dvc stage list --name-only"
    result = run_command(command)
    stages = [stage for stage in result.split("\n") if stage]
    logger.info(stages)
    return stages


########################################################################################
################################### PARAMETER ##########################################
from box import ConfigBox


@app.get("/Get-hyperparameter")
def get_hyperparameter():
    try :
        params = read_yaml(PARAMS_FILE_PATH)
        params_dict = params.to_dict()
        return params_dict
    except Exception as e:
        logger.error(f"Error while get hyperparameter: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@app.post("/Set-hyperparameter")
def set_hyperparameter(params: dict):
    try:
        # Use the provided dictionary directly
        params_dict = params
        # Convert Dictionary to ConfigBox
        paramsbox = ConfigBox(params_dict)
        write_yaml(PARAMS_FILE_PATH, paramsbox)
        return {"message": "Set hyperparameter successfully"}
    except Exception as e:
        logger.error(f"Error while set hyperparameter: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    
############################################################################################
################################### CONFIGURATION ##########################################
from box import ConfigBox


@app.get("/Get-configuration")
def get_configuration():
    try :
        configs = read_yaml(CONFIG_FILE_PATH)
        configs_dict = configs.to_dict()
        return configs_dict
    except Exception as e:
        logger.error(f"Error while get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@app.post("/Set-configuration")
def set_hyperparameter(configs: dict):
    try:
        # Use the provided dictionary directly
        configs_dict = configs
        # Convert Dictionary to ConfigBox
        configsbox = ConfigBox(configs_dict)
        write_yaml(CONFIG_FILE_PATH, configsbox)
        return {"message": "Set configuration successfully"}
    except Exception as e:
        logger.error(f"Error while set configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")


########################################################################################
###################################### GIT #############################################
# Check Git Tag
def git_get_tags():
    try:
        branch_commit = git_repo.commit(BRANCH_NAME)
        # tags = [tag for tag in git_repo.tags
        #         if tag.commit in branch_commit.iter_parents()
        #         or tag.commit == branch_commit]
        tags = [tag for tag in git_repo.tags
                if branch_commit in tag.commit.iter_parents() or
                tag.commit == branch_commit]
        return tags
    except Exception:
        pass

# Checkout Commit (Version)
def git_checkout_commit(commit_hash: str):
    if git_repo.is_dirty():
        message = "Repo has uncommitted changes"
        logger.warning(message)
        return False
    else:
        try:
            git_repo.git.checkout(commit_hash)
            message = f"Commit currently at: {get_current_commit_hash()}"
            logger.info(message)
            return True
        except Exception as e:
            logger.error(f"Error during checkout commit: {e}")
            return False

# Commit to git
def git_changes_commit(message: str):
    if git_repo.is_dirty():
        git_repo.git.add(A=True)
        git_repo.index.commit(message)
        logger.info("Commit created!")
        return True
    else:
        logger.info("No change to commit.")
        return False
    
# Remove changes
def git_remove_changes():
    try:
        # Log a warning before discarding changes
        logger.warning("You are about to discard all changes. Proceeding with checkout.")
        git_repo.git.checkout("--", ".")
        logger.info("Changes removed!")
        return True
    except Exception as e:
        logger.error(f"Error while remove changes: {e}")
        return False

def git_create_tag(tag_name: str, commit_hash: str):
    try:
        git_repo.create_tag(tag_name, commit_hash)
        logger.info(f"Tag '{tag_name}' was created at commit {commit_hash}")
        return True
    except Exception as e:
        logger.error(f"Error while create tag: {e}")
        return False

def git_detele_tag(tag_name: str):
    try:
        git_repo.delete_tag(tag_name)
        logger.info(f"Tag '{tag_name}' was deleted")
        return True
    except Exception as e:
        logger.error(f"Error while delete tag: {e}")
        return False    

def get_current_commit_hash():
    return git_repo.head.commit.hexsha

def get_current_branch():
    if git_repo.head.is_detached:
        logger.info("HEAD detached!")
        return None
    else:
        return git_repo.active_branch.name

def create_branch(branch_name: str):
    try:
        if branch_name in git_repo.branches:
            logger.info(f"Branch '{branch_name}' already exists")
        else:
            git_repo.git.checkout("-b", branch_name)
            logger.info(f"Branch '{branch_name}' was created")
        return True
    except Exception as e:
        logger.error(f"Error while create branch: {e}")
        return False
    
def delete_branch(branch_name: str):
    try:
        git_repo.git.branch("-D", branch_name)
        logger.info(f"Branch '{branch_name}' was deleted")
        return True
    except Exception as e:
        logger.error(f"Error while delete branch: {e}")
        return False
    
def rebase_branch(branch_name: str):
    try:
        git_repo.git.rebase(branch_name)
        logger.info(f"Branch '{branch_name}' was rebased")
        return True
    except GitCommandError as e:
        logger.error(f"Rebase error: {e}")

        # Nếu có lỗi và cần bỏ qua commit hiện tại
        try:
            git_repo.git.rebase("--skip")
            logger.info("Skipped commit and continued rebase.")
            return True
        except GitCommandError as e:
            logger.error(f"Failed to skip commit: {e}")
            return False


#########################################################################################
###################################### VERSION ##########################################
class Version():
    def __init__(self, name, hash, author, date, message):
        self.name = name
        self.hash = hash
        self.author = author
        self.date = date
        self.message = message

    def to_dict(self):
        return {
            "name": self.name,
            "hash": self.hash,
            "author": self.author,
            "date": self.date,
            "message": self.message
        }


class SaveVersionRequest(BaseModel):
    name: str
    message: str

@app.post("/Save-version")
def save_current_version(request: SaveVersionRequest):
    new_version_name = get_local_version()
    ret_push = dvc_push_data()
    # Create a tmp branch to save the version
    ret_cre_branch = create_branch("tmp-branch")
    ret_commit = git_changes_commit(request.message)
    # ret1_tag = git_create_tag(new_version_name, get_current_commit_hash())
    tag_name = request.name.replace(" ", "")
    tag_name = tag_name.replace("_", "")
    ret2_tag = git_create_tag(tag_name, get_current_commit_hash())# Tag with custom name
    ret_checkout = git_checkout_commit(BRANCH_NAME)
    ret_del_branch = delete_branch("tmp-branch")

    if ret_push and ret_cre_branch and ret_commit and ret_del_branch and ret_checkout and ret2_tag:
        message = "Save version successfully"
        logger.info(message)
        git_checkout_commit(tag_name)
        dvc_pull_data()
        return {"message": message}
    else:
        message = "Error while Save version"
        logger.error(message)
        raise HTTPException(status_code=500, detail=message)

class LoadVersionRequest(BaseModel):
    commit_hash: str
    load_artifacts: bool

@app.post("/Load-version")
def load_version(request: LoadVersionRequest):
    ret_checkout = git_checkout_commit(request.commit_hash)
    ret_pull = True
    if request.load_artifacts == True:
        ret_pull = dvc_pull_data()
    if ret_checkout and ret_pull:
        message = "Load version successfully"
        logger.info(message)
        return {"message": message}
    else:
        message = "Error while Load version"
        logger.error(message)
        raise HTTPException(status_code=500, detail=message)

@app.get("/Get-version")
def get_version():
    commit = git_repo.head.commit
    tags = [tag.name for tag in git_repo.tags if tag.commit == commit]
    # current_version = tags[-1] if len(tags) > 0 else None
    logger.info(f"List Tags: {tags}")
    return Version(tags,
                   commit.hexsha,
                   commit.author.name,
                   commit.committed_datetime.isoformat(),
                   commit.message.strip()).to_dict()

@app.post("/Ignore-version")
def ignore_version():
    ret_remove = git_remove_changes()
    ret_pull = dvc_pull_data()
    if ret_remove and ret_pull:
        message = "Ignore version successfully"
        logger.info(message)
        return {"message": message}
    else:
        message = "Error while Ignore version"
        logger.error(message)
        raise HTTPException(status_code=500, detail=message)

@app.get("/Checking-changes-vesion")
def checking_changes_version():
    try:
        unstaged_files = [item.a_path for item in git_repo.index.diff(None)]
        logger.info("Files modified but not staged: {}".format(unstaged_files))
        return unstaged_files
    except Exception as e:
        logger.error(f"Error while checking changes version: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.get("/Get-list-version")
def get_list_version():
    tags = git_get_tags()
    print(tags)
    if tags == None:
        message = "Error while get list version"
        logger.error(message)
        raise HTTPException(status_code=500, detail=message)
    # Get all tags and group them by commit hash
    commits = {}
    for tag in tags:
        commit = tag.commit
        if commit.hexsha not in commits:
            commits[commit.hexsha] = {
                'hash': commit.hexsha,
                'author': commit.author.name,
                'date': commit.committed_datetime.isoformat(),
                'message': commit.message.strip(),
                'tags': []
            }
        commits[commit.hexsha]['tags'].append(tag.name)
    print(commits)
    return [Version(data['tags'], hash, data['author'], data['date'], data['message'])
            for hash, data in commits.items()]


#########################################################################################
##################################### PREDICTION ########################################
from src.Classification.utils.common import create_directories
from src.Classification.pipeline.prediction import PredictionPipeline

# Constants
TEMP_FOLDER_PATH = os.path.join("artifacts", "predict_tmp")
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff", "image/webp", "image/gif"]

@app.post('/Prediction')
async def prediction(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        logger.error("Only image files are supported")
        return {"error": "Only image files are supported"}
    try:
        # save image to temp folder
        create_directories([TEMP_FOLDER_PATH])
        tmp_imgfile = os.path.join(TEMP_FOLDER_PATH, file.filename) 
        with open(tmp_imgfile, "wb") as buffer:
            while chunk := await file.read(8192):
                buffer.write(chunk)  
        # prediction
        output = PredictionPipeline(tmp_imgfile).predict()
        logger.info(output)
        return output
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    

#########################################################################################
##################################### DATASET ###########################################

# Root directory containing train, val, test folders
base_path = "./artifacts/data_ingestion/dataset"

def get_dataset_structure(base_path):
    dataset_structure = {}
    # Iterate through main directories (train, val, test)
    for main_dir in ["train", "val", "test"]:
        main_path = os.path.join(base_path, main_dir)
        sub_folder_dict = {}

        if os.path.exists(main_path):
            # Iterate through subdirectories
            for sub_folder in os.listdir(main_path):
                sub_path = os.path.join(main_path, sub_folder)

                if os.path.isdir(sub_path):  # Check if it is a directory
                    images = [
                        file for file in os.listdir(sub_path)
                        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'))
                    ]
                    if images:
                        sub_folder_dict[sub_folder] = images

        dataset_structure[main_dir] = sub_folder_dict
    return dataset_structure

@app.get('/Get-datasets-path')
def get_datasets():
    try:
        dataset_structure = get_dataset_structure(base_path)
        logger.info("Dataset structure retrieved successfully")
        return dataset_structure
    except Exception as e:
        logger.error(f"Error while retrieving datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    finally:
        dvc_repo.close()


###########################################################################################
##################################### ARTIFACTS ###########################################
@app.get("/Get-Count-Datasets")
def get_count_datasets():
    try:
        return FileResponse(COUNT_DATASET_PATH)
    except Exception as e:
        logger.error(f"Error while get count datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    
@app.get("/Get-Training-Graph")
def get_training_graph():
    try:
        return FileResponse(TRAINING_GRAPH_PATH)
    except Exception as e:
        logger.error(f"Error while get training graph: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.get("/Get-Confusion-Matrix")
def get_confusion_matrix():
    try:
        return FileResponse(CONFUSION_MATRIX_PATH)
    except Exception as e:
        logger.error(f"Error while get confusion matrix: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    
@app.get("/Get-Training-Log", response_class=PlainTextResponse)
async def get_training_log():
    try:
        with open(TRAINING_LOG_PATH, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        return str(e)

@app.get("/Get-Datasets")
def get_dataset():
    try:
        return FileResponse(DATASET_PATH)
    except Exception as e:
        logger.error(f"Error while get datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    
@app.get("/Get-ONNX-Model")
def get_onnx_model():
    try:
        return FileResponse(ONNX_MODEL_PATH)
    except Exception as e:
        logger.error(f"Error while get ONNX model: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    

###########################################################################################
##################################### SYSTEM INFO #########################################
from src.SystemInfo import *
@app.get("/Get-System-Information")
def get_system_information():
    try:
        cpu_info = get_cpu_info()
        memory_info = get_memory_info()
        disk_info = get_disk_info()
        gpu_info = get_gpu_info()
        system_info = get_system_info()
        return {"cpu_info": cpu_info,
                "memory_info": memory_info,
                "disk_info": disk_info,
                "gpu_info": gpu_info,
                "system_info": system_info}
    
    except Exception as e:
        logger.error(f"Error while get system information: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

########################################################################################
###################################### MAIN ############################################

if __name__ == "__main__":
    create_branch(BRANCH_NAME)
    # dvc_pull_data()
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn.run("webservice:app", host="0.0.0.0", port=8000, reload=True)
