# ASURA Framework

**WARNING:** Before we get started, please notice ASURA is NOT a clinical software/prototype. Repository data is for demonstration and reproducibility purposes ONLY.

The ASURA (**A**utomatic **S**kin **U**lcer **R**egion **A**ssessment) is a framework to assess the area of skin ulcer wounds through photographic images. By processing measurement tools, such as, rulers and metric tapes, ASURA is able to estimate the area in real world units.

## Usage

To test ASURA segmentation, you can use the `run.py` script. The parameters of the script is as follows:
 - ACTION: choose which script to run 
	- `train`
	-  `test`
	-  `evaluate`
	-  `augmentation` 
 - MODEL: see the list of models in json.settings
 - DATASET: see the list of datasets in json.settings
	 - The dataset must be as the `sample` dataset
	 - The `kfolds.csv` file indicates the folds of each image in the cross-validation
 - KFOLD: indicates which fold of the dataset to run

## License and Citation Request

The ASURA framework is a direct result of the following paper: 

> Chino, D. Y. T; Scabora, L. C.; Cazzolato, M. T.; Jorge, A. E. S.; Traina-Jr, C.; Traina, A. M. J.; *"Segmenting Skin Ulcers and Measuring the Wound Area Using Deep Convolutional Networks"* 

The ASURA's source code is available for researches and data scientists under the GNU General Public License. In case of publication and/or public use of the available sources, one should acknowledge its creators by citing the paper.
<!---
Bibtex:
```
@article{ChinoEtAl,  
      author = {Daniel Y. T. Chino and  
                Lucas C. Scabora and  	
    		    Mirela T. Cazzolato and 
                Ana E. S. Jorge and  
                Caetano Traina-Jr. and  
                Agma J. M. Traina},
      title = {Segmenting Skin Ulcers and Measuring the Wound Area Using Deep Convolutional Networks},
}
```
-->
