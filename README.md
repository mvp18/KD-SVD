Knowledge Distillation for Singing Voice Detection
================================================================

<h4>
Soumava Paul, Gurunath Reddy M, K. Sreenivasa Rao and Partha Pratim Das
</br>
<span style="font-size: 14pt; color: #555555">
Indian Institute of Technology Kharagpur
</span>
</h4>
<hr>

**INTERSPEECH 2021** | [arXiv](https://arxiv.org/abs/2011.04297) | [proceedings](https://www.isca-speech.org/archive/interspeech_2021/paul21b_interspeech.html)

## Setup

For dataset download, environment setup and data preparation, please refer to [this repo](https://github.com/kyungyunlee/ismir2018-revisiting-svd).

## Training and Testing

Refer to the following folders for reproducing results in the paper:

[1] Tables 2-4: [schluter-cnn](https://github.com/mvp18/KD-SVD/tree/master/schluter-cnn)
[2] Tables 5,6: [leglaive_lstm](https://github.com/mvp18/KD-SVD/tree/master/leglaive_lstm)
[3] Table 7: [lstm_scnn_feat](https://github.com/mvp18/KD-SVD/tree/master/lstm_scnn_feat)
[4] Table 8: [enkd_scnn_feat_student-cnn](https://github.com/mvp18/KD-SVD/tree/master/enkd_scnn_feat_student-cnn) and [enkd_scnn_feat_student-lstm](https://github.com/mvp18/KD-SVD/tree/master/enkd_scnn_feat_student-lstm)

Inside each folder, run `main.py` for baselines and `main_kd.py` for kd expts.
See `expts.sh` for sample runs.
Check `results` folder to get hyperparameter configs corresponding to highest validation accuracy. The corresponding test metrics are reported in our paper.

## 🎓 Cite

If this code was helpful for your research, consider citing:

```bibtex
@inproceedings{paul21b_interspeech,
  author={Soumava Paul and Gurunath Reddy M and K. Sreenivasa Rao and Partha Pratim Das},
  title={{Knowledge Distillation for Singing Voice Detection}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={4159--4163},
  doi={10.21437/Interspeech.2021-636}
}
```

## 🙏 Acknowledgements

We thank [Kyungyun Lee](https://github.com/kyungyunlee) for her [revisiting-svd](https://github.com/kyungyunlee/ismir2018-revisiting-svd) repo which proved to be the starting point of our work.