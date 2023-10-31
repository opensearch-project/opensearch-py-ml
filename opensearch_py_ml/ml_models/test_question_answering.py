# Save our model as pt
from question_answering_model import QuestionAnsweringModel
model_id = "distilbert-base-cased-distilled-squad"
folder_path = "question-model-folder"
our_pre_trained_model = QuestionAnsweringModel(model_id=model_id, folder_path=folder_path, overwrite=True)
zip_file_path = our_pre_trained_model.save_as_pt(model_id=model_id, sentences=["for example providing a small sentence", "we can add multiple sentences"])

# Obtain pytorch's official model
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
official_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

# List of questions to test
questions = ["Who was Jim Henson?", "Where do I live?", "What's my name?"]
contexts = ["Jim Henson was a nice puppet", "My name is Sarah and I live in London", "My name is Clara and I live in Berkeley."]

for i in range(len(questions)):
    question = questions[i]
    context = contexts[i]
    inputs = tokenizer(question, context, return_tensors="pt")
    print(f"=== test {i}, question: {question}, context: {context}")
    
    # Get official model's answer
    with torch.no_grad():
        outputs = official_model(**inputs)
    answer_start_index = torch.argmax(outputs.start_logits, dim=-1).item()
    answer_end_index = torch.argmax(outputs.end_logits, dim=-1).item()
    predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
    official_answer = tokenizer.decode(predict_answer_tokens)

    # Get our traced model's answer
    our_answer = our_pre_trained_model.test_traced_model(model_path=f"{folder_path}/{model_id}.pt", question=question, context=context)


    print(f"    Official answer: {official_answer}")
    print(f"    Our answer: {our_answer}")
