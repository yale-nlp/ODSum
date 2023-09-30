# ODSum

## Overview

ODSum introduces a benchmark for the task of Open Domain Multi-Document Summarization. This work is a result of collaborative research between Zhejiang University and Yale University.

The full details of the benchmarks and methodologies can be found in the [official paper](https://arxiv.org/pdf/2309.08960.pdf) authored by Yijie Zhou, Kejian Shi, Wencai Zhang, Yixin Liu, Yilun Zhao, and Arman Cohan.

## Dataset

The ODSum dataset is designed to evaluate the performance of modern summarization models in multi-document contexts spanning an open domain.

### Dataset Statistics

<img src="img/dataset statistics.png" alt="image-20230930180308855" style="zoom:50%;" />

### Dataset Structure

(Description of the dataset structure. This would typically include format descriptions, number of data points, any specific domain areas it covers, etc.)

## Models

[TODO]

## Data Processing

To process the data and convert it into formats compatible with various summarization models, refer to `data_process.ipynb` (or the appropriate processing notebook/script provided).

## Experimental Results

Retrieval Peformance

<img src="./img/RetPer.png" alt="image-20230930180329172" style="zoom:50%;" />



Summarization Performance for ODSum-story

![image-20230930180356879](./img/SumPerStory.png)



Summarization Performance for ODSum-meeting

![image-20230930180422951](./img/SumPerMeeting.png)

## Citation

If you use the ODSum dataset in your work, please kindly cite the following:

```
@article{zhou2023odsum,
   title={ODSUM: New Benchmarks for Open Domain Multi-Document Summarization},
   author={Zhou, Yijie and Shi, Kejian and Zhang, Wencai and Liu, Yixin and Zhao, Yilun and Cohan, Arman},
   journal={arXiv preprint arXiv:2309.08960},
   year={2023}
}
```

## Contacts

For any queries or issues related to ODSum, please contact:
- Yijie Zhou: e.j.zhou@zju.edu.cn
- Kejian Shi: kejian.shi@yale.edu
