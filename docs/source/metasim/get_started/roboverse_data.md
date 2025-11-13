# 💾 RoboVerse Dataset

We store all large trajectory files and assets on [HuggingFace](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data).

## When Manual Download is Not Required?

You do not need to download the full dataset if you only need to:

1. Try RoboVerse and run example code
2. Develop your own workflow
3. Run existing examples and workflows (an automatic download service is configured that will download required data on-demand during runtime)


## When to Download the Full Dataset?

Please download the complete dataset if you:

1. Want the full experience of RoboVerse's existing datasets and workflows
2. Prefer to avoid downloading data during runtime
3. Have sufficient storage space and time (the complete dataset exceeds 25 GB)
 


# Download the full RoboVerse dataset

Please follow the following steps.

### 1. Ensure Git and Git LFS are Installed

Install Git and Git LFS on your system if they are not already available:

```bash
sudo apt update
sudo apt install git git-lfs
git lfs install
```

### 2. Clone the Dataset Repository

Clone the dataset repository from Hugging Face:

```bash
git clone https://huggingface.co/datasets/RoboVerseOrg/roboverse_data
cd roboverse_data
```

### 3. Pull Large Files with Git LFS

If large files are not downloaded automatically, run the following:

```bash
git lfs pull
```

This process will create a local copy of the entire RoboVerse dataset repository on your machine.
