# EDiffuRec: An Enhanced Diffusion Model for Sequential Recommendation

This repository provides the official implementation of **EDiffuRec**, proposed in:

> Hanbyul Lee and Junghyun Kim,  
> "EDiffuRec: An Enhanced Diffusion Model for Sequential Recommendation,"  
> *Mathematics*, vol. 12(12), no. 1795, pp. 1-14, Jun. 2024. (https://doi.org/10.3390/math12121795)

EDiffuRec is based on and extends [DiffuRec](https://github.com/WHUIR/DiffuRec) (MIT License).  
We thank the original authors for releasing their code.

---

## Project Structure
```plaintext
EDiffuRec/
├── datasets/
│ └── data/ # Preprocessed datasets (e.g., dataset.pkl)
├── src/ # Source code
│ ├── data_preprocessing.py
│ ├── diffurec_hb_g.py
│ ├── main_hb.py
│ ├── model_hb.py
│ ├── step_sample.py
│ ├── trainer.py
│ └── utils.py
├── LICENSE # MIT License
└── README.md
```

### 1. Data Preparation

Preprocessed dataset files (e.g., dataset.pkl) should be placed under:

```bash
datasets/data/{dataset_name}/ 
```

### 2. Training and Evaluation
```bash
cd src
python main_hb.py --dataset amazon_beauty --noise_dist weibull --epochs 500
```

## Citation
If you use this code, please cite:

```bibtext
@article{lee2024ediffurec,
  title={EDiffuRec: An Enhanced Diffusion Model for Sequential Recommendation},
  author={Lee, Hanbyul and Kim, Junghyun},
  journal={Mathematics},
  volume={12},
  number={12},
  article-number={1795},
  pages={1--14},
  year={2024},
  publisher={MDPI}
}
```

## License
This project includes code from DiffuRec, which is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.


