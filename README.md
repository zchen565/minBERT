# minBERT

参考 CMU 11_711 Fall 2021 (Advanced NLP) Assignment2

包含预训练BERT（迷你版）和下游任务微调

只是个demo, 未使用gpu训练

## 文件
`_basic.py`: 一些nlp的基础函数(nltk)

`_util.py`: 辅助函数

`data.py`: 文件处理读取，torch数据集写入

`bert.py`: 模型

`config.py`: 模型较为小，就不写了

`train.py`: 训练

`SNLI.py`: 下游任务示例

## DataSet
https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/

## Dependency
只需要 torch (无需nltk等)

## Model
小中小中小

## Pretrain Loss
只是下降，没有实际价值

## Downstream Task
文本分类
注意：虽然预训练任务存了BERT模型，但下游任务使用别人训练好的模型（否则结果基本不能看）


