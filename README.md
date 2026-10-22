# Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods

This respository contains the code for the paper with the above name which was published and presented at ICML 2026. The paper is available in the Proceedings of Machine Learning Research [here](https://proceedings.mlr.press/v306/storck26a.html), see [Citation and License](#citation-and-license). 

## Getting Started

For the installation clone or download the repository and install the requirements:

```
pip install -r requirements.txt
```

**Reproducibility:** `./results` contains the results from the quantitative evaluation in `.json` format. These are created by running the different `./src/evaluate_*` scripts. `./imgs` contains the visualizations created by using `./src/plot_*` to create the visualizations shown in the paper. `./src/models` contains the implementation of all benchmarked models, `./src/utils` the implementations of the different fairness measures (and other helpers). Datasets are either downloaded or directly provided.

**Using the fair kernel decomposition:** `./src/models/decomposition_models.py` contains the implementation. The class `FairKernelDecomposition` contains the main logic (Algorithm 1). The class `Regularizer` is an interface that takes an arbitrary kernel based model and executes the fair decomposition step before fitting the kernel model with respect to the transformed kernel. Custom regularized models can then be easily created such as `RegularizedKRR` found therein. The function `nystroem_inverse()` performs the nystroem approximation of the kernel inverse (Algorithm 2, Appendix).


## Citation and License
```
@InProceedings{pmlr-v306-storck26a,
  title = 	 {Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods},
  author =       {St\"{o}rck, Felix and Hinder, Fabian and Hammer, Barbara},
  booktitle = 	 {Proceedings of the 43rd International Conference on Machine Learning},
  pages = 	 {115812--115825},
  year = 	 {2026},
  editor = 	 {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume = 	 {306},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--11 Jul},
  publisher =    {PMLR},
}
```

This project is licensed under the terms of the MIT license. For more details, see the [LICENSE](LICENSE) file.
