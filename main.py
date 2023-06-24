from predictionLSTM import getPredictedPrices
from firebase_admin import credentials, initialize_app, storage
import base64


cred = credentials.Certificate("./tradepal-698d2-firebase-adminsdk-jk207-322a5a9429.json")
initialize_app(cred, {
    'storageBucket': "tradepal-698d2.appspot.com"
})
def generateData():
    #Apple
    AAPL = getPredictedPrices("AAPL")
    AAPL.to_csv('./AAPL.csv')
    bucket = storage.bucket()
    blob = bucket.blob('AAPL.csv')
    blob.upload_from_filename('AAPL.csv')
    #Microsoft
    MSFT = getPredictedPrices("MSFT")
    MSFT.to_csv('./MSFT.csv')
    bucket = storage.bucket()
    blob = bucket.blob('MSFT.csv')
    blob.upload_from_filename('MSFT.csv')
    #Amazon
    AMZN = getPredictedPrices("AMZN")
    AMZN.to_csv('./AMZN.csv')
    bucket = storage.bucket()
    blob = bucket.blob('AMZN.csv')
    blob.upload_from_filename('AMZN.csv')
    #Tesla
    TSLA = getPredictedPrices("TSLA")
    TSLA.to_csv('./TSLA.csv')
    bucket = storage.bucket()
    blob = bucket.blob('TSLA.csv')
    blob.upload_from_filename('TSLA.csv')
    #Meta
    META = getPredictedPrices("META")
    META.to_csv('./META.csv')
    bucket = storage.bucket()
    blob = bucket.blob('META.csv')
    blob.upload_from_filename('META.csv')
    #Netflix
    NFLX = getPredictedPrices("NFLX")
    NFLX.to_csv('./NFLX.csv')
    bucket = storage.bucket()
    blob = bucket.blob('NFLX.csv')
    blob.upload_from_filename('NFLX.csv')
    #Alphabet (Google)
    GOOG = getPredictedPrices("GOOG")
    GOOG.to_csv('./GOOG.csv')
    bucket = storage.bucket()
    blob = bucket.blob('GOOG.csv')
    blob.upload_from_filename('GOOG.csv')
    #Intel
    INTC = getPredictedPrices("INTC")
    INTC.to_csv('./INTC.csv')
    bucket = storage.bucket()
    blob = bucket.blob('INTC.csv')
    blob.upload_from_filename('INTC.csv')
    #Nvidia
    NVDA = getPredictedPrices("NVDA")
    NVDA.to_csv('./NVDA.csv')
    bucket = storage.bucket()
    blob = bucket.blob('NVDA.csv')
    blob.upload_from_filename('NVDA.csv')
    #Pfizer
    PFE = getPredictedPrices("PFE")
    PFE.to_csv('./PFE.csv')
    bucket = storage.bucket()
    blob = bucket.blob('PFE.csv')
    blob.upload_from_filename('PFE.csv')
    #Dell
    DELL = getPredictedPrices("DELL")
    DELL.to_csv('./DELL.csv')
    bucket = storage.bucket()
    blob = bucket.blob('DELL.csv')
    blob.upload_from_filename('DELL.csv')
    #Warner Bros
    WBD = getPredictedPrices("WBD")
    WBD.to_csv('./WBD.csv')
    bucket = storage.bucket()
    blob = bucket.blob('WBD.csv')
    blob.upload_from_filename('WBD.csv')
    #Paypal
    PYPL = getPredictedPrices("PYPL")
    PYPL.to_csv('./PYPL.csv')
    bucket = storage.bucket()
    blob = bucket.blob('PYPL.csv')
    blob.upload_from_filename('PYPL.csv')
    #Oracle
    ORCL = getPredictedPrices("ORCL")
    ORCL.to_csv('./ORCL.csv')
    bucket = storage.bucket()
    blob = bucket.blob('ORCL.csv')
    blob.upload_from_filename('ORCL.csv')
    #Disney
    DIS = getPredictedPrices("DIS")
    DIS.to_csv('./DIS.csv')
    bucket = storage.bucket()
    blob = bucket.blob('DIS.csv')
    blob.upload_from_filename('DIS.csv')
    #Exxon Mobil
    XOM = getPredictedPrices("XOM")
    XOM.to_csv('./XOM.csv')
    bucket = storage.bucket()
    blob = bucket.blob('XOM.csv')
    blob.upload_from_filename('XOM.csv')
    #Cisco
    CSCO = getPredictedPrices("CSCO")
    CSCO.to_csv('./CSCO.csv')
    bucket = storage.bucket()
    blob = bucket.blob('CSCO.csv')
    blob.upload_from_filename('CSCO.csv')

def hello_pubsub(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    print(pubsub_message)
    generateData()