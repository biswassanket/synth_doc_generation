<p align="center">
  <img src="https://github.com/biswassanket/synth_doc_generation/blob/main/images/Teaser_ICDAR_2021-1.png">
  <br>
  <br>
  <b><i>The official GitHub repository for the paper on <a href="https://arxiv.org/abs/2107.02638">DocSynth: A Layout Guided Approach for Controllable Document Image Synthesis</a></i></b>
</p>

<p align="center">
  
#### The work will feature in <a href="https://icdar2021.org/">16th ICDAR 2021 (Lausanne, Switzerland)</a>.

### Getting Started
  

#### Step 1: Clone this repository and change directory to repository root
```bash
git clone https://github.com/biswassanket/synth_doc_generation.git
cd synth_doc_generation
```
#### Step 2: Make sure you have conda installed. If you do not have conda, here's the magic command to install miniconda.
```bash
curl -o ./miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./miniconda.sh
./miniconda.sh -b -u -p .
```
  
#### Step 3: Create an environment to run the project and install required dependencies.
* To create **conda** environment: `conda env create -f environment.yml`

#### Step 4: Activate the conda environment 
```bash
conda activate layout2im
```

#### Step 5: Downloading Dataset
* To download **PubLayNet** dataset: `curl -o <YOUR_TARGET_DIR>/publaynet.tar.gz https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz`

#### Step 5: Downloading the trained models
* Download the trained models to `checkpoints/pretrained/`. 
1. [trained model on PubLayNet](https://drive.google.com/file/d/1_yaxaXEINqAD-jlSqxtz1yIsor_0Lwam/view?usp=sharing)
  
#### Step 6: Testing 
  
Testing on PubLayNet dataset:
```bash
$ python layout2im/test.py --dataset publaynet --coco_dir datasets/publaynet \
                           --saved_model checkpoints/pretrained/publaynet_netG.pkl \
                           --results_dir checkpoints/pretrained_results_publaynet
```
#### Step 7: Training
```bash
$ python layout2im/train.py
```
### Results

#### 1. T-SNE Visualization of Synthetic Document Images

<p align='center'><img src='https://github.com/biswassanket/synth_doc_generation/blob/main/images/tsne_grid_synthetic.png' width='1000px'></p>

#### 2. Diverse results generated from the same document layout

<p align='center'><img src='https://github.com/biswassanket/synth_doc_generation/blob/main/images/results_diversity-1.png' width='1000px'></p>

#### 3. Examples of interactive generation of documents by the user 

<p align='center'><img src='https://github.com/biswassanket/synth_doc_generation/blob/main/images/results_add_bbox_ref-1.png' width='1000px'></p>
  
### Citation

If you find this code useful in your research then please cite
```
@inproceedings{biswas2021docsynth,
  title={DocSynth: A Layout Guided Approach for Controllable Document Image Synthesis},
  author={Biswas, Sanket and Riba, Pau and Llad{\'o}s, Josep and Pal, Umapada},
  booktitle={International Conference on Document Analysis and Recognition (ICDAR)},
  year={2021}
}
```
### Acknowledgement 
Our project has adapted and borrowed the code structure from [layout2im](https://github.com/zhaobozb/layout2im). 
We thank the authors. This research has been partially supported by the Spanish projects RTI2018-095645-B-C21, and FCT-19-15244, and the Catalan projects 2017-SGR-1783, the CERCA Program / Generalitat de Catalunya and PhD Scholarship from AGAUR (2021FIB-10010).
  
### Authors
* [Sanket Biswas](https://github.com/biswassanket)
* [Pau Riba](https://github.com/priba)
  
### Conclusion
Thank you and sorry for the bugs!
