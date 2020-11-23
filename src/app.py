import sparknlp
from pyspark.sql.types import StringType
from flask import Flask, jsonify, request
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql import functions as sf


# creating a Flask app
app = Flask(__name__)


@app.route('/home/<str>', methods=['GET'])
def return_prediction(str):
    return make_prediction(str)


def make_prediction(string):

    spark = sparknlp.start()

    dfTest = spark.createDataFrame([string], StringType()).toDF("text")

    pretrained_pipeline = PretrainedPipeline("analyze_sentimentdl_use_twitter")

    preds = (
        pretrained_pipeline
        .transform(dfTest)
        .withColumn("result", sf.col("sentiment.result").getItem(0))
        .select("result")
    ).toPandas()["result"].values.tolist()[0]

    return preds


if __name__ == '__main__':

    app.run(debug=True)
