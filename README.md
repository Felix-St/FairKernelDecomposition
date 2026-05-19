# Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods

This respository contains the code for the paper with the above name. 
The paper is available on arxiv [https://arxiv.org/abs/2511.03304], see [Citation and License](#citation-and-license). 

## Getting Started

For the installation clone or download the repository and install the requirements:

```
pip install -r requirements.txt
```

**Reproducibility:** ./results contains the results from the quantiative evaluation in '.json' format. These are created by running the different ./src/evaluate_* scripts. ./imgs contains the visualizations created by using ./src/plot_* to create the visualizations shown in the paper. ./src/models contains the implementation of all benchmarked models, ./src/utils the implementations of the different fairness measures (and other helpers). Datasets are either downloaded or directly provided.

**Using the fair kernel decomposition:** ./src/models/decomposition_models.py contains the implementation. The class `FairKernelDecomposition` contains the main logic (Algorithm 1). The class `Regularizer` is an interface that takes an arbitrary kernel based model and executes the fair decomposition step before fitting the kernel model with respect to the transformed kernel. Custom regularized models can then be easily created such as `RegularizedKRR` found therein. The function `nystroem_inverse()` performs the nystroem approximation of the kernel inverse (Algorithm 2, Appendix).


## Citation and License

This work has been accepted at ICML 2026 and the information below will be updated soon.

Please cite the arxiv preprint if you are using this repository:

```
@misc{störck2025extendingfairnullspaceprojections,
      title={Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods}, 
      author={Felix Störck and Fabian Hinder and Barbara Hammer},
      year={2025},
      eprint={2511.03304},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.03304}, 
}
```

This project is licensed under the terms of the MIT license. For more details, see the [LICENSE](LICENSE) file.
