1. Item Ratings (ratings.txt): [user-id, item-id, rating-value]

2. Trust Ratings (trust.txt):  [user-id (trustor), user-id (trustee), trust-value]

The trust links are directed. 

3. To use this data set in your research, please consider to cite our work: 

@INPROCEEDINGS{guo2013novel,
  author = {Guo, G. and Zhang, J. and Yorke-Smith, N.},
  title = {A Novel Bayesian Similarity Measure for Recommender Systems},
  booktitle = {Proceedings of the 23rd International Joint Conference on Artificial Intelligence (IJCAI)},
  year = {2013},
  pages = {2619-2625}
}

FilmTrust 是从真实系统中爬取的数据集，包含用户对电影的评分数据以及用户间的信任关系。
在 FilmTrust 中，用户给电影的评分范围是 0.5 分到 4 分，共八个等级。
用户与用户之间的关系是一个二元关系，即1 表示信任关系，0 表示没有信任关系。
数据集中包含有 1508 个用户对 2071 个项目的 35497 个评分信息以及 1632 条用户之间的信任关系信息