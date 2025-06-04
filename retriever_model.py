import torch
import torch.nn.functional as F

class Retriever:
    def __init__(self, all_questions, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device
        self.all_questions = all_questions
        self.question_embeddings = self.get_embedding(all_questions)  # compute once

    def get_embedding(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0]
        return F.normalize(cls_embeddings, p=2, dim=1)

    def get_examples(self, question, k=5):
        question_embedding = self.get_embedding([question])
        similarities = torch.matmul(question_embedding, self.question_embeddings.T).squeeze(0)
        topk_indices = similarities.topk(k).indices.tolist()
        return [self.all_questions[i] for i in topk_indices]
