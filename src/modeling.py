import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import (
    BertSentenceEmbeddings,
    SentimentDLApproach,
    UniversalSentenceEncoder,
    ClassifierDLApproach
)
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql import functions as sf
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from sklearn.metrics import accuracy_score, recall_score, precision_score
import time
import pandas as pd

from src.helper_funs import *
from config import *


def get_train_test(n, data_path, spark):

    data = spark.read.json(data_path).limit(n)

    # Transform dataframe to text and binary label
    data = (
      data
      .withColumn(
          "label",
          sf.when(data.stars >= 4, 1.0).otherwise(0.0)
      )
      .select("text", "label")
    )

    # Under sample positive reviews to balance classes
    positive_count = data.filter(data["label"] == 1.0).count()
    negative_count = n - positive_count
    positive_sample_perc = negative_count / positive_count

    positive_sample = data.filter(data["label"] == 1.0).sample(False, positive_sample_perc, seed=0)
    data_balanced = data.filter(data["label"] == 0.0).union(positive_sample)

    # Train test split
    train, test = data_balanced.randomSplit([0.8, 0.2], seed=421)

    return train, test


def create_bert_encoding_pipeline():

    document = (
      DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    )

    bert_sent = (
      BertSentenceEmbeddings
      .pretrained()
      .setInputCols(["document"])
      .setOutputCol("sentence_embeddings")
    )

    sentimentdl = (
      SentimentDLApproach()
      .setInputCols(["sentence_embeddings"])
      .setOutputCol("class")
      .setLabelColumn("label")
      .setMaxEpochs(5)
      .setEnableOutputLogs(True)
      .setThreshold(0.5)
    )

    pipeline = Pipeline(
      stages=[
          document,
          bert_sent,
          sentimentdl
      ]
    )

    return pipeline


def create_universal_pipeline():

    document = (
      DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    )

    use = (
      UniversalSentenceEncoder.pretrained()
      .setInputCols(["document"])
      .setOutputCol("sentence_embeddings")
    )

    classsifierdl = (
      ClassifierDLApproach()
      .setInputCols(["sentence_embeddings"])
      .setOutputCol("class")
      .setLabelColumn("label")
      .setMaxEpochs(5)
      .setEnableOutputLogs(True)
    )

    pipeline = Pipeline(
      stages=[
          document,
          use,
          classsifierdl
      ]
    )

    return pipeline


def fit_model(pipeline, train, test, model_description):

    start_time = time.time()

    pipelineModel = pipeline.fit(train)

    preds_df = (
      pipelineModel
      .transform(test)
      .withColumn(
          "prediction",
          sf.col("class.result").getItem(0).cast(DoubleType())
      )
      .select("label", "prediction")
    ).toPandas()

    predictions_save_path = os.path.join(
      OUTPUT_PATH,
      f"predictions_{model_description}.csv"
    )
    preds_df.to_csv(predictions_save_path, index=False)

    accuracy = accuracy_score(preds_df["label"], preds_df["prediction"])
    precision = precision_score(preds_df["label"], preds_df["prediction"])
    recall = recall_score(preds_df["label"], preds_df["prediction"])

    eval_end_time = time.time()

    results_dict = {
      "model description": model_description,
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "time": eval_end_time - start_time,
    }

    results_save_path = os.path.join(
      OUTPUT_PATH,
      f"{model_description}.json"
    )
    save_dict_as_json(results_dict, results_save_path)

    return pipelineModel


if __name__ == '__main__':

    spark = sparknlp.start()

    # Generate balanced train/test
    train, test = get_train_test(
        n=50000,
        data_path=DATA_PATH,
        spark=spark
    )

    # Fit and evaluate model
    pipeline = create_universal_pipeline()
    model = fit_model(pipeline, train, test, "final_model")

    # Save model
    model.write().overwrite().save(MODEL_PATH)
