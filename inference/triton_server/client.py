import tritonclient.http as http_client
from tritonclient.utils import *
import numpy as np

ENABLE_SSL = False
ENDPOINT_URL = 'localhost:8000'
HTTP_HEADERS = {"Authorization": "Bearer __PASTE_KEY_HERE__"}

# Connect to the server
if ENABLE_SSL:
    import gevent.ssl
    triton_http_client = http_client.InferenceServerClient(
        url=ENDPOINT_URL, verbose=False,
        ssl=True, ssl_context_factory=gevent.ssl._create_default_https_context,
    )
else:
    triton_http_client = http_client.InferenceServerClient(
        url=ENDPOINT_URL, verbose=False,
    )

print("Is server ready - {}".format(triton_http_client.is_server_ready(headers=HTTP_HEADERS)))

def get_string_tensor(string_values, tensor_name):
    string_obj = np.array(string_values, dtype="object")
    input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
    input_obj.set_data_from_numpy(string_obj)
    return input_obj

def get_translation_input_for_triton(texts: list, src_lang: str, tgt_lang: str):
    return [
        get_string_tensor([[text] for text in texts], "INPUT_TEXT"),
        get_string_tensor([[src_lang]] * len(texts), "INPUT_LANGUAGE_ID"),
        get_string_tensor([[tgt_lang]] * len(texts), "OUTPUT_LANGUAGE_ID"),
    ]

# Prepare input and output tensors
input_sentences = ["Hello world, I am Ram and I am from Ayodhya.", "How are you Ravan bro?"]
inputs = get_translation_input_for_triton(input_sentences, "en", "hi")
output0 = http_client.InferRequestedOutput("OUTPUT_TEXT")

# Send request
response = triton_http_client.infer(
    "nmt",
    model_version='1',
    inputs=inputs,
    outputs=[output0],
    headers=HTTP_HEADERS,
)#.get_response()

# Decode the response
output_batch = response.as_numpy('OUTPUT_TEXT').tolist()
for input_sentence, translation in zip(input_sentences, output_batch):
    print()
    print(input_sentence)
    print(translation[0].decode("utf-8"))
