
from opensearch_py_ml.ml_models import QuestionAnsweringModel

# Save our model as pt or onnx
model_id = "distilbert-base-cased-distilled-squad"
folder_path = "question-model-folder"
our_pre_trained_model = QuestionAnsweringModel(model_id=model_id, folder_path=folder_path, overwrite=True)
# zip_file_path = our_pre_trained_model.save_as_pt(model_id=model_id, sentences=["for example providing a small sentence", "we can add multiple sentences"])
zip_file_path = our_pre_trained_model.save_as_onnx(model_id=model_id)

# List of questions to test
questions = ["Who was Jim Henson?", "Where do I live?", "What's my name?"]
contexts = ["Jim Henson was a nice puppet", "My name is Sarah and I live in London", "My name is Clara and I live in Berkeley."]

# Obtain pytorch's official model
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
official_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

def official_model_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = official_model(**inputs)
    answer_start_index = torch.argmax(outputs.start_logits, dim=-1).item()
    answer_end_index = torch.argmax(outputs.end_logits, dim=-1).item()
    predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
    official_answer = tokenizer.decode(predict_answer_tokens)
    return official_answer

def test_onnx():
    from transformers import AutoTokenizer
    from onnxruntime import InferenceSession
    import numpy as np
    session = InferenceSession(f"{folder_path}/{model_id}.onnx")

    for i in range(len(questions)):
        question = questions[i]
        context = contexts[i]
        inputs = tokenizer(question, context, return_tensors="pt")
        print(f"=== test {i}, question: {question}, context: {context}")

        inputs = tokenizer(question, context, return_tensors="np")
        outputs = session.run(output_names=["start_logits", "end_logits"], input_feed=dict(inputs))

        answer_start_index = np.argmax(outputs[0], axis=-1).item()
        answer_end_index = np.argmax(outputs[1], axis=-1).item()
        predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens)

        print(f"    Official answer: {official_model_answer(question, context)}")
        print(f"    Our answer: {answer}")
    
def test_pt():
    traced_model = torch.jit.load(f"{folder_path}/{model_id}.pt")

    for i in range(len(questions)):
        question = questions[i]
        context = contexts[i]
        inputs = tokenizer(question, context, return_tensors="pt")
        print(f"=== test {i}, question: {question}, context: {context}")

        with torch.no_grad():
            outputs = traced_model(**inputs)
        answer_start_index = torch.argmax(outputs["start_logits"], dim=-1).item()
        answer_end_index = torch.argmax(outputs["end_logits"], dim=-1).item()
        predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens)

        print(f"    Official answer: {official_model_answer(question, context)}")
        print(f"    Our answer: {answer}")

test_onnx()