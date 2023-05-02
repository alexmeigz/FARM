# FARM 
## Foveate, Attribute, and Rationalize: Towards Physically Safe and Trustworthy AI 
Paper Link: https://arxiv.org/abs/2212.09667 
Authors: Alex Mei*, Sharon Levy, William Yang Wang

## Setup
- Install venv: https://realpython.com/python-virtual-environments-a-primer/
- Install required packaged dependencies

## Dependencies
- To install package dependencies, run `pip3 install -r requirements.txt`
- To update package dependencies, run `pip3 freeze > requirements.txt`

## Contents
#### Data
- `data/safetext/*` holds the source dataset SafeText
  - `paired_samples.json` contains the original SafeText dataset
  - `safe_samples.json` and `unsafe_samples.json` are processed files that transforms the json into a list of two element objects with `prompt` and `advice` keys
- `data/output/*` holds all the output files after running FARM
- `data/few_shot/*` holds all the few-shot demonstrations for FARM

#### Modules
- `models/eval.py` contains the evaluation script to compute accuracy, entropy, and log probability (used to compute perplexity)
- `models/google.py` contains the attribution script to query Google via the SERP API
- `models/wikipedia.py` contains the attribution script to query Wikipedia API
- `models/gpt.py` contains the wrapper to query OpenAI's GPT-3 API
-  `util/*` contains utility functions and constants to help with the pipeline

#### Main 
- `main.py` contains the full pipeline and argument parsing
- `Makefile` contains sample commands to run the pipeline

## Environment Variables
Add a `.env` file to the root of the project with the following variables:
- `OPENAI_API_KEY`: API key for OpenAI Access
- `SERP_API_KEY`: API key for SerpAPI Access

## Usage
- Create a new venv with `python3 -m venv .venv`
- Activate venv with `source .venv/bin/activate`

```
usage: main.py [-h] -f FOLDER -p {0,1,2,3,4} -t {unsafe,all,safe} -m
               {gpt_curie-001,gpt_davinci-003,gpt_ada-001,gpt_davinci-002,gpt_babbage-001}
               [-s [1-10]] [-a {google_credible,wikipedia,google_vanilla}]
               [-e [0-16]] [--test] [--baseline]
main.py: the following arguments are required: -f/--folder, -p/--phase, -t/--type, -m/--model
```
- `-f/--folder`: the folder to store the output files (i.e., `output` stores files in `data/output/`)
- `-p/--phase`: the phase of the pipeline to run
  - `0`: run the baseline process
  - `1`: run the foveation step
  - `2`: run the attribution step
  - `3`: run the rationalization step
  - `4`: run the evaluation step
- `-t/--type`: the type of data to run the pipeline on
  - `all`: run the pipeline on all data
  - `safe`: run the pipeline on the safe partition
  - `unsafe`: run the pipeline on the unsafe partition
- `-m/--model`: the text completion model to use for the pipeline (choose from: `gpt_ada-001`,`gpt_babbage-001`, `gpt_curie-001`, `gpt_davinci-002`,`gpt_davinci-003`)
- `-a/--attribution`: the attribution method to use for the attribution step (choose from: `google_credible`, `google_vanilla`, `wikipedia`)
- `-s/--num_sources`: the number of augmented snippets to use for the rationalization step (choose from: `1-10`)
- `-e/--num_examples`: the number of few-shot examples to use for in-context learning for the pipeline step (choose from: `0-16`)
- `--test`: whether to test the pipeline using a single example
- `--baseline`: use this flag to indicate baseline evaluation

## Attribution
When using resources based on our project, please cite the following paper, to appear in ACL 2023:
```
@misc{mei2022foveate,
      title={Foveate, Attribute, and Rationalize: Towards Safe and Trustworthy AI}, 
      author={Alex Mei and Sharon Levy and William Yang Wang},
      year={2022},
      eprint={2212.09667},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```