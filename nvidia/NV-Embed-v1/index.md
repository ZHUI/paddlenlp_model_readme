
# NV-Embed-v1
---


## README([From Huggingface](https://huggingface.co/nvidia/NV-Embed-v1))

---
tags:
- mteb
- sentence-transformers
model-index:
- name: NV-Embed-v1
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en)
      config: en
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 95.11940298507461
    - type: ap
      value: 79.21521293687752
    - type: f1
      value: 92.45575440759485
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_polarity
      name: MTEB AmazonPolarityClassification
      config: default
      split: test
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
    metrics:
    - type: accuracy
      value: 97.143125
    - type: ap
      value: 95.28635983806933
    - type: f1
      value: 97.1426073127198
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (en)
      config: en
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 55.465999999999994
    - type: f1
      value: 52.70196166254287
  - task:
      type: Retrieval
    dataset:
      type: mteb/arguana
      name: MTEB ArguAna
      config: default
      split: test
      revision: c22ab2a51041ffd869aaddef7af8d8215647e41a
    metrics:
    - type: map_at_1
      value: 44.879000000000005
    - type: map_at_10
      value: 60.146
    - type: map_at_100
      value: 60.533
    - type: map_at_1000
      value: 60.533
    - type: map_at_3
      value: 55.725
    - type: map_at_5
      value: 58.477999999999994
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 44.879000000000005
    - type: ndcg_at_10
      value: 68.205
    - type: ndcg_at_100
      value: 69.646
    - type: ndcg_at_1000
      value: 69.65599999999999
    - type: ndcg_at_3
      value: 59.243
    - type: ndcg_at_5
      value: 64.214
    - type: precision_at_1
      value: 44.879000000000005
    - type: precision_at_10
      value: 9.374
    - type: precision_at_100
      value: 0.996
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 23.139000000000003
    - type: precision_at_5
      value: 16.302
    - type: recall_at_1
      value: 44.879000000000005
    - type: recall_at_10
      value: 93.741
    - type: recall_at_100
      value: 99.57300000000001
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 69.417
    - type: recall_at_5
      value: 81.50800000000001
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-p2p
      name: MTEB ArxivClusteringP2P
      config: default
      split: test
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
    metrics:
    - type: v_measure
      value: 53.76391569504432
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-s2s
      name: MTEB ArxivClusteringS2S
      config: default
      split: test
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
    metrics:
    - type: v_measure
      value: 49.589284930659005
  - task:
      type: Reranking
    dataset:
      type: mteb/askubuntudupquestions-reranking
      name: MTEB AskUbuntuDupQuestions
      config: default
      split: test
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
    metrics:
    - type: map
      value: 67.49860736554155
    - type: mrr
      value: 80.77771182341819
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_pearson
      value: 87.87900681188576
    - type: cos_sim_spearman
      value: 85.5905044545741
    - type: euclidean_pearson
      value: 86.80150192033507
    - type: euclidean_spearman
      value: 85.5905044545741
    - type: manhattan_pearson
      value: 86.79080500635683
    - type: manhattan_spearman
      value: 85.69351885001977
  - task:
      type: Classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77Classification
      config: default
      split: test
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
    metrics:
    - type: accuracy
      value: 90.33766233766235
    - type: f1
      value: 90.20736178753944
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-p2p
      name: MTEB BiorxivClusteringP2P
      config: default
      split: test
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
    metrics:
    - type: v_measure
      value: 48.152262077598465
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-s2s
      name: MTEB BiorxivClusteringS2S
      config: default
      split: test
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
    metrics:
    - type: v_measure
      value: 44.742970683037235
  - task:
      type: Retrieval
    dataset:
      type: mteb/cqadupstack
      name: MTEB CQADupstackRetrieval
      config: default
      split: test
      revision: 46989137a86843e03a6195de44b09deda022eec7
    metrics:
    - type: map_at_1
      value: 31.825333333333326
    - type: map_at_10
      value: 44.019999999999996
    - type: map_at_100
      value: 45.37291666666667
    - type: map_at_1000
      value: 45.46991666666666
    - type: map_at_3
      value: 40.28783333333333
    - type: map_at_5
      value: 42.39458333333334
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 37.79733333333333
    - type: ndcg_at_10
      value: 50.50541666666667
    - type: ndcg_at_100
      value: 55.59125
    - type: ndcg_at_1000
      value: 57.06325
    - type: ndcg_at_3
      value: 44.595666666666666
    - type: ndcg_at_5
      value: 47.44875
    - type: precision_at_1
      value: 37.79733333333333
    - type: precision_at_10
      value: 9.044083333333333
    - type: precision_at_100
      value: 1.3728333333333336
    - type: precision_at_1000
      value: 0.16733333333333333
    - type: precision_at_3
      value: 20.842166666666667
    - type: precision_at_5
      value: 14.921916666666668
    - type: recall_at_1
      value: 31.825333333333326
    - type: recall_at_10
      value: 65.11916666666666
    - type: recall_at_100
      value: 86.72233333333335
    - type: recall_at_1000
      value: 96.44200000000001
    - type: recall_at_3
      value: 48.75691666666667
    - type: recall_at_5
      value: 56.07841666666666
  - task:
      type: Retrieval
    dataset:
      type: mteb/climate-fever
      name: MTEB ClimateFEVER
      config: default
      split: test
      revision: 47f2ac6acb640fc46020b02a5b59fdda04d39380
    metrics:
    - type: map_at_1
      value: 14.698
    - type: map_at_10
      value: 25.141999999999996
    - type: map_at_100
      value: 27.1
    - type: map_at_1000
      value: 27.277
    - type: map_at_3
      value: 21.162
    - type: map_at_5
      value: 23.154
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 32.704
    - type: ndcg_at_10
      value: 34.715
    - type: ndcg_at_100
      value: 41.839
    - type: ndcg_at_1000
      value: 44.82
    - type: ndcg_at_3
      value: 28.916999999999998
    - type: ndcg_at_5
      value: 30.738
    - type: precision_at_1
      value: 32.704
    - type: precision_at_10
      value: 10.795
    - type: precision_at_100
      value: 1.8530000000000002
    - type: precision_at_1000
      value: 0.241
    - type: precision_at_3
      value: 21.564
    - type: precision_at_5
      value: 16.261
    - type: recall_at_1
      value: 14.698
    - type: recall_at_10
      value: 41.260999999999996
    - type: recall_at_100
      value: 65.351
    - type: recall_at_1000
      value: 81.759
    - type: recall_at_3
      value: 26.545999999999996
    - type: recall_at_5
      value: 32.416
  - task:
      type: Retrieval
    dataset:
      type: mteb/dbpedia
      name: MTEB DBPedia
      config: default
      split: test
      revision: c0f706b76e590d620bd6618b3ca8efdd34e2d659
    metrics:
    - type: map_at_1
      value: 9.959
    - type: map_at_10
      value: 23.104
    - type: map_at_100
      value: 33.202
    - type: map_at_1000
      value: 35.061
    - type: map_at_3
      value: 15.911
    - type: map_at_5
      value: 18.796
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 63.5
    - type: ndcg_at_10
      value: 48.29
    - type: ndcg_at_100
      value: 52.949999999999996
    - type: ndcg_at_1000
      value: 60.20100000000001
    - type: ndcg_at_3
      value: 52.92
    - type: ndcg_at_5
      value: 50.375
    - type: precision_at_1
      value: 73.75
    - type: precision_at_10
      value: 38.65
    - type: precision_at_100
      value: 12.008000000000001
    - type: precision_at_1000
      value: 2.409
    - type: precision_at_3
      value: 56.083000000000006
    - type: precision_at_5
      value: 48.449999999999996
    - type: recall_at_1
      value: 9.959
    - type: recall_at_10
      value: 28.666999999999998
    - type: recall_at_100
      value: 59.319
    - type: recall_at_1000
      value: 81.973
    - type: recall_at_3
      value: 17.219
    - type: recall_at_5
      value: 21.343999999999998
  - task:
      type: Classification
    dataset:
      type: mteb/emotion
      name: MTEB EmotionClassification
      config: default
      split: test
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
    metrics:
    - type: accuracy
      value: 91.705
    - type: f1
      value: 87.98464515154814
  - task:
      type: Retrieval
    dataset:
      type: mteb/fever
      name: MTEB FEVER
      config: default
      split: test
      revision: bea83ef9e8fb933d90a2f1d5515737465d613e12
    metrics:
    - type: map_at_1
      value: 74.297
    - type: map_at_10
      value: 83.931
    - type: map_at_100
      value: 84.152
    - type: map_at_1000
      value: 84.164
    - type: map_at_3
      value: 82.708
    - type: map_at_5
      value: 83.536
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 80.048
    - type: ndcg_at_10
      value: 87.77000000000001
    - type: ndcg_at_100
      value: 88.467
    - type: ndcg_at_1000
      value: 88.673
    - type: ndcg_at_3
      value: 86.003
    - type: ndcg_at_5
      value: 87.115
    - type: precision_at_1
      value: 80.048
    - type: precision_at_10
      value: 10.711
    - type: precision_at_100
      value: 1.1320000000000001
    - type: precision_at_1000
      value: 0.117
    - type: precision_at_3
      value: 33.248
    - type: precision_at_5
      value: 20.744
    - type: recall_at_1
      value: 74.297
    - type: recall_at_10
      value: 95.402
    - type: recall_at_100
      value: 97.97
    - type: recall_at_1000
      value: 99.235
    - type: recall_at_3
      value: 90.783
    - type: recall_at_5
      value: 93.55499999999999
  - task:
      type: Retrieval
    dataset:
      type: mteb/fiqa
      name: MTEB FiQA2018
      config: default
      split: test
      revision: 27a168819829fe9bcd655c2df245fb19452e8e06
    metrics:
    - type: map_at_1
      value: 32.986
    - type: map_at_10
      value: 55.173
    - type: map_at_100
      value: 57.077
    - type: map_at_1000
      value: 57.176
    - type: map_at_3
      value: 48.182
    - type: map_at_5
      value: 52.303999999999995
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 62.037
    - type: ndcg_at_10
      value: 63.096
    - type: ndcg_at_100
      value: 68.42200000000001
    - type: ndcg_at_1000
      value: 69.811
    - type: ndcg_at_3
      value: 58.702
    - type: ndcg_at_5
      value: 60.20100000000001
    - type: precision_at_1
      value: 62.037
    - type: precision_at_10
      value: 17.269000000000002
    - type: precision_at_100
      value: 2.309
    - type: precision_at_1000
      value: 0.256
    - type: precision_at_3
      value: 38.992
    - type: precision_at_5
      value: 28.610999999999997
    - type: recall_at_1
      value: 32.986
    - type: recall_at_10
      value: 70.61800000000001
    - type: recall_at_100
      value: 89.548
    - type: recall_at_1000
      value: 97.548
    - type: recall_at_3
      value: 53.400000000000006
    - type: recall_at_5
      value: 61.29599999999999
  - task:
      type: Retrieval
    dataset:
      type: mteb/hotpotqa
      name: MTEB HotpotQA
      config: default
      split: test
      revision: ab518f4d6fcca38d87c25209f94beba119d02014
    metrics:
    - type: map_at_1
      value: 41.357
    - type: map_at_10
      value: 72.91499999999999
    - type: map_at_100
      value: 73.64699999999999
    - type: map_at_1000
      value: 73.67899999999999
    - type: map_at_3
      value: 69.113
    - type: map_at_5
      value: 71.68299999999999
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 82.714
    - type: ndcg_at_10
      value: 79.92
    - type: ndcg_at_100
      value: 82.232
    - type: ndcg_at_1000
      value: 82.816
    - type: ndcg_at_3
      value: 74.875
    - type: ndcg_at_5
      value: 77.969
    - type: precision_at_1
      value: 82.714
    - type: precision_at_10
      value: 17.037
    - type: precision_at_100
      value: 1.879
    - type: precision_at_1000
      value: 0.196
    - type: precision_at_3
      value: 49.471
    - type: precision_at_5
      value: 32.124
    - type: recall_at_1
      value: 41.357
    - type: recall_at_10
      value: 85.18599999999999
    - type: recall_at_100
      value: 93.964
    - type: recall_at_1000
      value: 97.765
    - type: recall_at_3
      value: 74.207
    - type: recall_at_5
      value: 80.31099999999999
  - task:
      type: Classification
    dataset:
      type: mteb/imdb
      name: MTEB ImdbClassification
      config: default
      split: test
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
    metrics:
    - type: accuracy
      value: 97.05799999999998
    - type: ap
      value: 95.51324940484382
    - type: f1
      value: 97.05788617110184
  - task:
      type: Retrieval
    dataset:
      type: mteb/msmarco
      name: MTEB MSMARCO
      config: default
      split: test
      revision: c5a29a104738b98a9e76336939199e264163d4a0
    metrics:
    - type: map_at_1
      value: 25.608999999999998
    - type: map_at_10
      value: 39.098
    - type: map_at_100
      value: 0
    - type: map_at_1000
      value: 0
    - type: map_at_3
      value: 0
    - type: map_at_5
      value: 37.383
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 26.404
    - type: ndcg_at_10
      value: 46.493
    - type: ndcg_at_100
      value: 0
    - type: ndcg_at_1000
      value: 0
    - type: ndcg_at_3
      value: 0
    - type: ndcg_at_5
      value: 42.459
    - type: precision_at_1
      value: 26.404
    - type: precision_at_10
      value: 7.249
    - type: precision_at_100
      value: 0
    - type: precision_at_1000
      value: 0
    - type: precision_at_3
      value: 0
    - type: precision_at_5
      value: 11.874
    - type: recall_at_1
      value: 25.608999999999998
    - type: recall_at_10
      value: 69.16799999999999
    - type: recall_at_100
      value: 0
    - type: recall_at_1000
      value: 0
    - type: recall_at_3
      value: 0
    - type: recall_at_5
      value: 56.962
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (en)
      config: en
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 96.50706794345645
    - type: f1
      value: 96.3983656000426
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (en)
      config: en
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 89.77428180574556
    - type: f1
      value: 70.47378359921777
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (en)
      config: en
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 80.07061197041023
    - type: f1
      value: 77.8633288994029
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (en)
      config: en
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 81.74176193678547
    - type: f1
      value: 79.8943810025071
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-p2p
      name: MTEB MedrxivClusteringP2P
      config: default
      split: test
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
    metrics:
    - type: v_measure
      value: 39.239199736486334
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-s2s
      name: MTEB MedrxivClusteringS2S
      config: default
      split: test
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
    metrics:
    - type: v_measure
      value: 36.98167653792483
  - task:
      type: Reranking
    dataset:
      type: mteb/mind_small
      name: MTEB MindSmallReranking
      config: default
      split: test
      revision: 3bdac13927fdc888b903db93b2ffdbd90b295a69
    metrics:
    - type: map
      value: 30.815595271130718
    - type: mrr
      value: 31.892823243368795
  - task:
      type: Retrieval
    dataset:
      type: mteb/nfcorpus
      name: MTEB NFCorpus
      config: default
      split: test
      revision: ec0fa4fe99da2ff19ca1214b7966684033a58814
    metrics:
    - type: map_at_1
      value: 6.214
    - type: map_at_10
      value: 14.393
    - type: map_at_100
      value: 18.163999999999998
    - type: map_at_1000
      value: 19.753999999999998
    - type: map_at_3
      value: 10.737
    - type: map_at_5
      value: 12.325
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 48.297000000000004
    - type: ndcg_at_10
      value: 38.035000000000004
    - type: ndcg_at_100
      value: 34.772
    - type: ndcg_at_1000
      value: 43.631
    - type: ndcg_at_3
      value: 44.252
    - type: ndcg_at_5
      value: 41.307
    - type: precision_at_1
      value: 50.15500000000001
    - type: precision_at_10
      value: 27.647
    - type: precision_at_100
      value: 8.824
    - type: precision_at_1000
      value: 2.169
    - type: precision_at_3
      value: 40.97
    - type: precision_at_5
      value: 35.17
    - type: recall_at_1
      value: 6.214
    - type: recall_at_10
      value: 18.566
    - type: recall_at_100
      value: 34.411
    - type: recall_at_1000
      value: 67.331
    - type: recall_at_3
      value: 12.277000000000001
    - type: recall_at_5
      value: 14.734
  - task:
      type: Retrieval
    dataset:
      type: mteb/nq
      name: MTEB NQ
      config: default
      split: test
      revision: b774495ed302d8c44a3a7ea25c90dbce03968f31
    metrics:
    - type: map_at_1
      value: 47.11
    - type: map_at_10
      value: 64.404
    - type: map_at_100
      value: 65.005
    - type: map_at_1000
      value: 65.01400000000001
    - type: map_at_3
      value: 60.831
    - type: map_at_5
      value: 63.181
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 52.983999999999995
    - type: ndcg_at_10
      value: 71.219
    - type: ndcg_at_100
      value: 73.449
    - type: ndcg_at_1000
      value: 73.629
    - type: ndcg_at_3
      value: 65.07
    - type: ndcg_at_5
      value: 68.715
    - type: precision_at_1
      value: 52.983999999999995
    - type: precision_at_10
      value: 10.756
    - type: precision_at_100
      value: 1.198
    - type: precision_at_1000
      value: 0.121
    - type: precision_at_3
      value: 28.977999999999998
    - type: precision_at_5
      value: 19.583000000000002
    - type: recall_at_1
      value: 47.11
    - type: recall_at_10
      value: 89.216
    - type: recall_at_100
      value: 98.44500000000001
    - type: recall_at_1000
      value: 99.744
    - type: recall_at_3
      value: 73.851
    - type: recall_at_5
      value: 82.126
  - task:
      type: Retrieval
    dataset:
      type: mteb/quora
      name: MTEB QuoraRetrieval
      config: default
      split: test
      revision: e4e08e0b7dbe3c8700f0daef558ff32256715259
    metrics:
    - type: map_at_1
      value: 71.641
    - type: map_at_10
      value: 85.687
    - type: map_at_100
      value: 86.304
    - type: map_at_1000
      value: 86.318
    - type: map_at_3
      value: 82.811
    - type: map_at_5
      value: 84.641
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 82.48
    - type: ndcg_at_10
      value: 89.212
    - type: ndcg_at_100
      value: 90.321
    - type: ndcg_at_1000
      value: 90.405
    - type: ndcg_at_3
      value: 86.573
    - type: ndcg_at_5
      value: 88.046
    - type: precision_at_1
      value: 82.48
    - type: precision_at_10
      value: 13.522
    - type: precision_at_100
      value: 1.536
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.95
    - type: precision_at_5
      value: 24.932000000000002
    - type: recall_at_1
      value: 71.641
    - type: recall_at_10
      value: 95.91499999999999
    - type: recall_at_100
      value: 99.63300000000001
    - type: recall_at_1000
      value: 99.994
    - type: recall_at_3
      value: 88.248
    - type: recall_at_5
      value: 92.428
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering
      name: MTEB RedditClustering
      config: default
      split: test
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
    metrics:
    - type: v_measure
      value: 63.19631707795757
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering-p2p
      name: MTEB RedditClusteringP2P
      config: default
      split: test
      revision: 385e3cb46b4cfa89021f56c4380204149d0efe33
    metrics:
    - type: v_measure
      value: 68.01353074322002
  - task:
      type: Retrieval
    dataset:
      type: mteb/scidocs
      name: MTEB SCIDOCS
      config: default
      split: test
      revision: f8c2fcf00f625baaa80f62ec5bd9e1fff3b8ae88
    metrics:
    - type: map_at_1
      value: 4.67
    - type: map_at_10
      value: 11.991999999999999
    - type: map_at_100
      value: 14.263
    - type: map_at_1000
      value: 14.59
    - type: map_at_3
      value: 8.468
    - type: map_at_5
      value: 10.346
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 23.1
    - type: ndcg_at_10
      value: 20.19
    - type: ndcg_at_100
      value: 28.792
    - type: ndcg_at_1000
      value: 34.406
    - type: ndcg_at_3
      value: 19.139
    - type: ndcg_at_5
      value: 16.916
    - type: precision_at_1
      value: 23.1
    - type: precision_at_10
      value: 10.47
    - type: precision_at_100
      value: 2.2849999999999997
    - type: precision_at_1000
      value: 0.363
    - type: precision_at_3
      value: 17.9
    - type: precision_at_5
      value: 14.979999999999999
    - type: recall_at_1
      value: 4.67
    - type: recall_at_10
      value: 21.21
    - type: recall_at_100
      value: 46.36
    - type: recall_at_1000
      value: 73.72999999999999
    - type: recall_at_3
      value: 10.865
    - type: recall_at_5
      value: 15.185
  - task:
      type: STS
    dataset:
      type: mteb/sickr-sts
      name: MTEB SICK-R
      config: default
      split: test
      revision: 20a6d6f312dd54037fe07a32d58e5e168867909d
    metrics:
    - type: cos_sim_pearson
      value: 84.31392081916142
    - type: cos_sim_spearman
      value: 82.80375234068289
    - type: euclidean_pearson
      value: 81.4159066418654
    - type: euclidean_spearman
      value: 82.80377112831907
    - type: manhattan_pearson
      value: 81.48376861134983
    - type: manhattan_spearman
      value: 82.86696725667119
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_pearson
      value: 84.1940844467158
    - type: cos_sim_spearman
      value: 76.22474792649982
    - type: euclidean_pearson
      value: 79.87714243582901
    - type: euclidean_spearman
      value: 76.22462054296349
    - type: manhattan_pearson
      value: 80.19242023327877
    - type: manhattan_spearman
      value: 76.53202564089719
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_pearson
      value: 85.58028303401805
    - type: cos_sim_spearman
      value: 86.30355131725051
    - type: euclidean_pearson
      value: 85.9027489087145
    - type: euclidean_spearman
      value: 86.30352515906158
    - type: manhattan_pearson
      value: 85.74953930990678
    - type: manhattan_spearman
      value: 86.21878393891001
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_pearson
      value: 82.92370135244734
    - type: cos_sim_spearman
      value: 82.09196894621044
    - type: euclidean_pearson
      value: 81.83198023906334
    - type: euclidean_spearman
      value: 82.09196482328333
    - type: manhattan_pearson
      value: 81.8951479497964
    - type: manhattan_spearman
      value: 82.2392819738236
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_pearson
      value: 87.05662816919057
    - type: cos_sim_spearman
      value: 87.24083005603993
    - type: euclidean_pearson
      value: 86.54673655650183
    - type: euclidean_spearman
      value: 87.24083428218053
    - type: manhattan_pearson
      value: 86.51248710513431
    - type: manhattan_spearman
      value: 87.24796986335883
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_pearson
      value: 84.06330254316376
    - type: cos_sim_spearman
      value: 84.76788840323285
    - type: euclidean_pearson
      value: 84.15438606134029
    - type: euclidean_spearman
      value: 84.76788840323285
    - type: manhattan_pearson
      value: 83.97986968570088
    - type: manhattan_spearman
      value: 84.52468572953663
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 88.08627867173213
    - type: cos_sim_spearman
      value: 87.41531216247836
    - type: euclidean_pearson
      value: 87.92912483282956
    - type: euclidean_spearman
      value: 87.41531216247836
    - type: manhattan_pearson
      value: 87.85418528366228
    - type: manhattan_spearman
      value: 87.32655499883539
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (en)
      config: en
      split: test
      revision: eea2b4fe26a775864c896887d910b76a8098ad3f
    metrics:
    - type: cos_sim_pearson
      value: 70.74143864859911
    - type: cos_sim_spearman
      value: 69.84863549051433
    - type: euclidean_pearson
      value: 71.07346533903932
    - type: euclidean_spearman
      value: 69.84863549051433
    - type: manhattan_pearson
      value: 71.32285810342451
    - type: manhattan_spearman
      value: 70.13063960824287
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_pearson
      value: 86.05702492574339
    - type: cos_sim_spearman
      value: 86.13895001731495
    - type: euclidean_pearson
      value: 85.86694514265486
    - type: euclidean_spearman
      value: 86.13895001731495
    - type: manhattan_pearson
      value: 85.96382530570494
    - type: manhattan_spearman
      value: 86.30950247235928
  - task:
      type: Reranking
    dataset:
      type: mteb/scidocs-reranking
      name: MTEB SciDocsRR
      config: default
      split: test
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
    metrics:
    - type: map
      value: 87.26225076335467
    - type: mrr
      value: 96.60696329813977
  - task:
      type: Retrieval
    dataset:
      type: mteb/scifact
      name: MTEB SciFact
      config: default
      split: test
      revision: 0228b52cf27578f30900b9e5271d331663a030d7
    metrics:
    - type: map_at_1
      value: 64.494
    - type: map_at_10
      value: 74.102
    - type: map_at_100
      value: 74.571
    - type: map_at_1000
      value: 74.58
    - type: map_at_3
      value: 71.111
    - type: map_at_5
      value: 73.184
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 67.667
    - type: ndcg_at_10
      value: 78.427
    - type: ndcg_at_100
      value: 80.167
    - type: ndcg_at_1000
      value: 80.41
    - type: ndcg_at_3
      value: 73.804
    - type: ndcg_at_5
      value: 76.486
    - type: precision_at_1
      value: 67.667
    - type: precision_at_10
      value: 10.167
    - type: precision_at_100
      value: 1.107
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 28.222
    - type: precision_at_5
      value: 18.867
    - type: recall_at_1
      value: 64.494
    - type: recall_at_10
      value: 90.422
    - type: recall_at_100
      value: 97.667
    - type: recall_at_1000
      value: 99.667
    - type: recall_at_3
      value: 78.278
    - type: recall_at_5
      value: 84.828
  - task:
      type: PairClassification
    dataset:
      type: mteb/sprintduplicatequestions-pairclassification
      name: MTEB SprintDuplicateQuestions
      config: default
      split: test
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
    metrics:
    - type: cos_sim_accuracy
      value: 99.82772277227723
    - type: cos_sim_ap
      value: 95.93881941923254
    - type: cos_sim_f1
      value: 91.12244897959184
    - type: cos_sim_precision
      value: 93.02083333333333
    - type: cos_sim_recall
      value: 89.3
    - type: dot_accuracy
      value: 99.82772277227723
    - type: dot_ap
      value: 95.93886287716076
    - type: dot_f1
      value: 91.12244897959184
    - type: dot_precision
      value: 93.02083333333333
    - type: dot_recall
      value: 89.3
    - type: euclidean_accuracy
      value: 99.82772277227723
    - type: euclidean_ap
      value: 95.93881941923253
    - type: euclidean_f1
      value: 91.12244897959184
    - type: euclidean_precision
      value: 93.02083333333333
    - type: euclidean_recall
      value: 89.3
    - type: manhattan_accuracy
      value: 99.83366336633664
    - type: manhattan_ap
      value: 96.07286531485964
    - type: manhattan_f1
      value: 91.34912461380021
    - type: manhattan_precision
      value: 94.16135881104034
    - type: manhattan_recall
      value: 88.7
    - type: max_accuracy
      value: 99.83366336633664
    - type: max_ap
      value: 96.07286531485964
    - type: max_f1
      value: 91.34912461380021
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering
      name: MTEB StackExchangeClustering
      config: default
      split: test
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
    metrics:
    - type: v_measure
      value: 74.98877944689897
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering-p2p
      name: MTEB StackExchangeClusteringP2P
      config: default
      split: test
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
    metrics:
    - type: v_measure
      value: 42.0365286267706
  - task:
      type: Reranking
    dataset:
      type: mteb/stackoverflowdupquestions-reranking
      name: MTEB StackOverflowDupQuestions
      config: default
      split: test
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
    metrics:
    - type: map
      value: 56.5797777961647
    - type: mrr
      value: 57.57701754944402
  - task:
      type: Summarization
    dataset:
      type: mteb/summeval
      name: MTEB SummEval
      config: default
      split: test
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
    metrics:
    - type: cos_sim_pearson
      value: 30.673216240991756
    - type: cos_sim_spearman
      value: 31.198648165051225
    - type: dot_pearson
      value: 30.67321511262982
    - type: dot_spearman
      value: 31.198648165051225
  - task:
      type: Retrieval
    dataset:
      type: mteb/trec-covid
      name: MTEB TRECCOVID
      config: default
      split: test
      revision: bb9466bac8153a0349341eb1b22e06409e78ef4e
    metrics:
    - type: map_at_1
      value: 0.23500000000000001
    - type: map_at_10
      value: 2.274
    - type: map_at_100
      value: 14.002
    - type: map_at_1000
      value: 34.443
    - type: map_at_3
      value: 0.705
    - type: map_at_5
      value: 1.162
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 88
    - type: ndcg_at_10
      value: 85.883
    - type: ndcg_at_100
      value: 67.343
    - type: ndcg_at_1000
      value: 59.999
    - type: ndcg_at_3
      value: 87.70400000000001
    - type: ndcg_at_5
      value: 85.437
    - type: precision_at_1
      value: 92
    - type: precision_at_10
      value: 91.2
    - type: precision_at_100
      value: 69.19999999999999
    - type: precision_at_1000
      value: 26.6
    - type: precision_at_3
      value: 92.667
    - type: precision_at_5
      value: 90.8
    - type: recall_at_1
      value: 0.23500000000000001
    - type: recall_at_10
      value: 2.409
    - type: recall_at_100
      value: 16.706
    - type: recall_at_1000
      value: 56.396
    - type: recall_at_3
      value: 0.734
    - type: recall_at_5
      value: 1.213
  - task:
      type: Retrieval
    dataset:
      type: mteb/touche2020
      name: MTEB Touche2020
      config: default
      split: test
      revision: a34f9a33db75fa0cbb21bb5cfc3dae8dc8bec93f
    metrics:
    - type: map_at_1
      value: 2.4819999999999998
    - type: map_at_10
      value: 10.985
    - type: map_at_100
      value: 17.943
    - type: map_at_1000
      value: 19.591
    - type: map_at_3
      value: 5.86
    - type: map_at_5
      value: 8.397
    - type: mrr_at_1
      value: 0
    - type: mrr_at_10
      value: 0
    - type: mrr_at_100
      value: 0
    - type: mrr_at_1000
      value: 0
    - type: mrr_at_3
      value: 0
    - type: mrr_at_5
      value: 0
    - type: ndcg_at_1
      value: 37.755
    - type: ndcg_at_10
      value: 28.383000000000003
    - type: ndcg_at_100
      value: 40.603
    - type: ndcg_at_1000
      value: 51.469
    - type: ndcg_at_3
      value: 32.562000000000005
    - type: ndcg_at_5
      value: 31.532
    - type: precision_at_1
      value: 38.775999999999996
    - type: precision_at_10
      value: 24.898
    - type: precision_at_100
      value: 8.429
    - type: precision_at_1000
      value: 1.582
    - type: precision_at_3
      value: 31.973000000000003
    - type: precision_at_5
      value: 31.019999999999996
    - type: recall_at_1
      value: 2.4819999999999998
    - type: recall_at_10
      value: 17.079
    - type: recall_at_100
      value: 51.406
    - type: recall_at_1000
      value: 84.456
    - type: recall_at_3
      value: 6.802
    - type: recall_at_5
      value: 10.856
  - task:
      type: Classification
    dataset:
      type: mteb/toxic_conversations_50k
      name: MTEB ToxicConversationsClassification
      config: default
      split: test
      revision: edfaf9da55d3dd50d43143d90c1ac476895ae6de
    metrics:
    - type: accuracy
      value: 92.5984
    - type: ap
      value: 41.969971606260906
    - type: f1
      value: 78.95995145145926
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_extraction
      name: MTEB TweetSentimentExtractionClassification
      config: default
      split: test
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
    metrics:
    - type: accuracy
      value: 80.63950198075835
    - type: f1
      value: 80.93345710055597
  - task:
      type: Clustering
    dataset:
      type: mteb/twentynewsgroups-clustering
      name: MTEB TwentyNewsgroupsClustering
      config: default
      split: test
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
    metrics:
    - type: v_measure
      value: 60.13491858535076
  - task:
      type: PairClassification
    dataset:
      type: mteb/twittersemeval2015-pairclassification
      name: MTEB TwitterSemEval2015
      config: default
      split: test
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
    metrics:
    - type: cos_sim_accuracy
      value: 87.42325803182929
    - type: cos_sim_ap
      value: 78.72789856051176
    - type: cos_sim_f1
      value: 71.83879093198993
    - type: cos_sim_precision
      value: 68.72289156626506
    - type: cos_sim_recall
      value: 75.25065963060686
    - type: dot_accuracy
      value: 87.42325803182929
    - type: dot_ap
      value: 78.72789755269454
    - type: dot_f1
      value: 71.83879093198993
    - type: dot_precision
      value: 68.72289156626506
    - type: dot_recall
      value: 75.25065963060686
    - type: euclidean_accuracy
      value: 87.42325803182929
    - type: euclidean_ap
      value: 78.7278973892869
    - type: euclidean_f1
      value: 71.83879093198993
    - type: euclidean_precision
      value: 68.72289156626506
    - type: euclidean_recall
      value: 75.25065963060686
    - type: manhattan_accuracy
      value: 87.59015318590929
    - type: manhattan_ap
      value: 78.99631410090865
    - type: manhattan_f1
      value: 72.11323565929972
    - type: manhattan_precision
      value: 68.10506566604127
    - type: manhattan_recall
      value: 76.62269129287598
    - type: max_accuracy
      value: 87.59015318590929
    - type: max_ap
      value: 78.99631410090865
    - type: max_f1
      value: 72.11323565929972
  - task:
      type: PairClassification
    dataset:
      type: mteb/twitterurlcorpus-pairclassification
      name: MTEB TwitterURLCorpus
      config: default
      split: test
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
    metrics:
    - type: cos_sim_accuracy
      value: 89.15473279776458
    - type: cos_sim_ap
      value: 86.05463278065247
    - type: cos_sim_f1
      value: 78.63797449855686
    - type: cos_sim_precision
      value: 74.82444552596816
    - type: cos_sim_recall
      value: 82.86110255620572
    - type: dot_accuracy
      value: 89.15473279776458
    - type: dot_ap
      value: 86.05463366261054
    - type: dot_f1
      value: 78.63797449855686
    - type: dot_precision
      value: 74.82444552596816
    - type: dot_recall
      value: 82.86110255620572
    - type: euclidean_accuracy
      value: 89.15473279776458
    - type: euclidean_ap
      value: 86.05463195314907
    - type: euclidean_f1
      value: 78.63797449855686
    - type: euclidean_precision
      value: 74.82444552596816
    - type: euclidean_recall
      value: 82.86110255620572
    - type: manhattan_accuracy
      value: 89.15861373074087
    - type: manhattan_ap
      value: 86.08743411620402
    - type: manhattan_f1
      value: 78.70125023325248
    - type: manhattan_precision
      value: 76.36706018686174
    - type: manhattan_recall
      value: 81.18263012011087
    - type: max_accuracy
      value: 89.15861373074087
    - type: max_ap
      value: 86.08743411620402
    - type: max_f1
      value: 78.70125023325248
language:
- en
license: cc-by-nc-4.0
---
## Introduction
We introduce NV-Embed, a generalist embedding model that ranks No. 1 on the Massive Text Embedding Benchmark ([MTEB benchmark](https://arxiv.org/abs/2210.07316))(as of May 24, 2024), with 56 tasks, encompassing retrieval, reranking, classification, clustering, and semantic textual similarity tasks. Notably, our model also achieves the highest score of 59.36 on 15 retrieval tasks within this benchmark.

NV-Embed presents several new designs, including having the LLM attend to latent vectors for better pooled embedding output, and demonstrating a two-stage instruction tuning method to enhance the accuracy of both retrieval and non-retrieval tasks.

For more technical details, refer to our paper: [NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models](https://arxiv.org/pdf/2405.17428).

For more benchmark results (other than MTEB), please find the [AIR-Bench](https://huggingface.co/spaces/AIR-Bench/leaderboard) for QA (English only) and Long-Doc.

## Model Details
- Base Decoder-only LLM: [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- Pooling Type: Latent-Attention
- Embedding Dimension: 4096

## How to use

Here is an example of how to encode queries and passages using Huggingface-transformer and Sentence-transformer. Please find the required package version [here](https://huggingface.co/nvidia/NV-Embed-v1#2-required-packages).

### Usage (HuggingFace Transformers)

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}

query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
queries = [
    'are judo throws allowed in wrestling?', 
    'how to become a radiology technician in michigan?'
    ]

# No instruction needed for retrieval passages
passage_prefix = ""
passages = [
    "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
    "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
]

# load model with tokenizer
model = AutoModel.from_pretrained('nvidia/NV-Embed-v1', trust_remote_code=True)

# get the embeddings
max_length = 4096
query_embeddings = model.encode(queries, instruction=query_prefix, max_length=max_length)
passage_embeddings = model.encode(passages, instruction=passage_prefix, max_length=max_length)

# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

# get the embeddings with DataLoader (spliting the datasets into multiple mini-batches)
# batch_size=2
# query_embeddings = model._do_encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
# passage_embeddings = model._do_encode(passages, batch_size=batch_size, instruction=passage_prefix, max_length=max_length, num_workers=32, return_numpy=True)

scores = (query_embeddings @ passage_embeddings.T) * 100
print(scores.tolist())
#[[77.9402084350586, 0.4248958230018616], [3.757718086242676, 79.60113525390625]]
```


### Usage (Sentence-Transformers)

```python
import torch
from sentence_transformers import SentenceTransformer

# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}

query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
queries = [
    'are judo throws allowed in wrestling?', 
    'how to become a radiology technician in michigan?'
    ]

# No instruction needed for retrieval passages
passages = [
    "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
    "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
]

# load model with tokenizer
model = SentenceTransformer('nvidia/NV-Embed-v1', trust_remote_code=True)
model.max_seq_length = 4096
model.tokenizer.padding_side="right"

def add_eos(input_examples):
  input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
  return input_examples

# get the embeddings
batch_size = 2
query_embeddings = model.encode(add_eos(queries), batch_size=batch_size, prompt=query_prefix, normalize_embeddings=True)
passage_embeddings = model.encode(add_eos(passages), batch_size=batch_size, normalize_embeddings=True)

scores = (query_embeddings @ passage_embeddings.T) * 100
print(scores.tolist())
```

## Correspondence to
Chankyu Lee (chankyul@nvidia.com), Rajarshi Roy (rajarshir@nvidia.com), Wei Ping (wping@nvidia.com)

## Citation
If you find this code useful in your research, please consider citing:

```bibtex
@misc{lee2024nvembed,
      title={NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models}, 
      author={Chankyu Lee and Rajarshi Roy and Mengyao Xu and Jonathan Raiman and Mohammad Shoeybi and Bryan Catanzaro and Wei Ping},
      year={2024},
      eprint={2405.17428},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## License
This model should not be used for any commercial purpose. Refer the [license](https://spdx.org/licenses/CC-BY-NC-4.0) for the detailed terms.

For commercial purpose, we recommend you to use the models of [NeMo Retriever Microservices (NIMs)](https://build.nvidia.com/explore/retrieval).


## Troubleshooting


#### 1. How to enable Multi-GPU (Note, this is the case for HuggingFace Transformers)
```python
from transformers import AutoModel
from torch.nn import DataParallel

embedding_model = AutoModel.from_pretrained("nvidia/NV-Embed-v1")
for module_key, module in embedding_model._modules.items():
    embedding_model._modules[module_key] = DataParallel(module)
```

#### 2. Required Packages

If you have trouble, try installing the python packages as below
```python
pip uninstall -y transformer-engine
pip install torch==2.2.0
pip install transformers==4.42.4
pip install flash-attn==2.2.0
pip install sentence-transformers==2.7.0
```

#### 3. Fixing "nvidia/NV-Embed-v1 is not the path to a directory containing a file named config.json"

Switch to your local model path，and open config.json and change the value of **"_name_or_path"** and replace it with your local model path.


#### 4. Access to model nvidia/NV-Embed-v1 is restricted. You must be authenticated to access it

Use your huggingface access [token](https://huggingface.co/settings/tokens) to execute *"huggingface-cli login"*.

#### 5. How to resolve slight mismatch in Sentence transformer results.

A slight mismatch in the Sentence Transformer implementation is caused by a discrepancy in the calculation of the instruction prefix length within the Sentence Transformer package.

To fix this issue, you need to build the Sentence Transformer package from source, making the necessary modification in this [line](https://github.com/UKPLab/sentence-transformers/blob/v2.7-release/sentence_transformers/SentenceTransformer.py#L353) as below.
```python
git clone https://github.com/UKPLab/sentence-transformers.git
cd sentence-transformers
git checkout v2.7-release
# Modify L353 in SentenceTransformer.py to **'extra_features["prompt_length"] = tokenized_prompt["input_ids"].shape[-1]'**.
pip install -e .
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/README.md) (51.3 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/config.json) (707.0 B)

- [model_state-00001-of-00004.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/model_state-00001-of-00004.pdparams) (3.7 GB)

- [model_state-00002-of-00004.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/model_state-00002-of-00004.pdparams) (3.7 GB)

- [model_state-00003-of-00004.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/model_state-00003-of-00004.pdparams) (3.7 GB)

- [model_state-00004-of-00004.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/model_state-00004-of-00004.pdparams) (3.6 GB)

- [model_state.pdparams.index.json](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/model_state.pdparams.index.json) (23.8 KB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/sentencepiece.bpe.model) (481.9 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/special_tokens_map.json) (72.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/tokenizer.json) (1.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/nvidia/NV-Embed-v1/tokenizer_config.json) (1.4 KB)


[Back to Main](../../)